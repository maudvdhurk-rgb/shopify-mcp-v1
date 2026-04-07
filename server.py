#!/usr/bin/env python3
"""
Shopify MCP Server — Full Admin API access via FastMCP.
Provides tools for managing products, orders, customers, collections,
inventory, fulfillments, and theme files through the Shopify Admin REST API.
"""
import json
import os
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_STORE        = os.environ.get("SHOPIFY_STORE", "")
SHOPIFY_TOKEN        = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_CLIENT_ID    = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
API_VERSION          = os.environ.get("SHOPIFY_API_VERSION", "2024-10")
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shopify_mcp")

PORT          = int(os.environ.get("PORT", "8000"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

mcp = FastMCP("shopify_mcp", host="0.0.0.0", port=PORT, json_response=True)


# ---------------------------------------------------------------------------
# Token Manager
# ---------------------------------------------------------------------------

class TokenManager:
    def __init__(self, store, client_id, client_secret, static_token="", refresh_buffer=1800):
        self._store         = store
        self._client_id     = client_id
        self._client_secret = client_secret
        self._static_token  = static_token
        self._refresh_buffer = refresh_buffer
        self._access_token: str   = ""
        self._expires_at: float   = 0.0
        self._lock = asyncio.Lock()
        self._use_client_credentials = bool(client_id and client_secret)

        if self._use_client_credentials:
            logger.info("Token mode: client_credentials (auto-refresh enabled)")
        elif static_token:
            logger.info("Token mode: static SHOPIFY_ACCESS_TOKEN (no auto-refresh)")
            self._access_token = static_token
            self._expires_at   = float("inf")
        else:
            logger.warning("No credentials configured.")

    @property
    def is_expired(self):
        if not self._access_token:
            return True
        return time.time() >= (self._expires_at - self._refresh_buffer)

    async def get_token(self):
        if not self.is_expired:
            return self._access_token
        async with self._lock:
            if not self.is_expired:
                return self._access_token
            if self._use_client_credentials:
                await self._refresh_token()
            elif not self._access_token:
                raise RuntimeError("No valid token available.")
        return self._access_token

    async def force_refresh(self):
        if not self._use_client_credentials:
            raise RuntimeError("Cannot refresh — using a static token.")
        async with self._lock:
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self):
        url = f"https://{self._store}.myshopify.com/admin/oauth/access_token"
        logger.info("Refreshing Shopify access token...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={"grant_type": "client_credentials", "client_id": self._client_id, "client_secret": self._client_secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Token refresh failed ({resp.status_code}).")
            data               = resp.json()
            self._access_token = data["access_token"]
            expires_in         = data.get("expires_in", 86399)
            self._expires_at   = time.time() + expires_in
            logger.info(f"Token refreshed. Expires in {expires_in}s.")


token_manager = TokenManager(
    store=SHOPIFY_STORE,
    client_id=SHOPIFY_CLIENT_ID,
    client_secret=SHOPIFY_CLIENT_SECRET,
    static_token=SHOPIFY_TOKEN,
    refresh_buffer=TOKEN_REFRESH_BUFFER,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_url():
    return f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}"

async def _headers():
    token = await token_manager.get_token()
    return {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}

async def _request(method, path, params=None, body=None, _retried=False):
    if not SHOPIFY_STORE:
        raise RuntimeError("Missing SHOPIFY_STORE environment variable.")
    url     = f"{_base_url()}/{path}"
    headers = await _headers()
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, headers=headers, params=params, json=body, timeout=30.0)
        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            await token_manager.force_refresh()
            return await _request(method, path, params=params, body=body, _retried=True)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

def _error(e):
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        messages = {
            401: "Authentication failed — check your token.",
            403: "Permission denied — token may be missing required scopes.",
            404: "Resource not found.",
            422: f"Validation error: {json.dumps(detail)}",
            429: "Rate-limited — wait a moment and retry.",
        }
        return messages.get(status, f"Shopify API error {status}: {json.dumps(detail)}")
    if isinstance(e, httpx.TimeoutException):
        return "Request timed out."
    if isinstance(e, RuntimeError):
        return str(e)
    return f"Unexpected error: {type(e).__name__}: {e}"

def _fmt(data):
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════

class ListProductsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int]  = Field(default=50, ge=1, le=250)
    status:         Optional[str]  = Field(default=None)
    product_type:   Optional[str]  = Field(default=None)
    vendor:         Optional[str]  = Field(default=None)
    collection_id:  Optional[int]  = Field(default=None)
    since_id:       Optional[int]  = Field(default=None)
    fields:         Optional[str]  = Field(default=None)

@mcp.tool(name="shopify_list_products", annotations={"readOnlyHint": True})
async def shopify_list_products(params: ListProductsInput) -> str:
    """List products from the Shopify store with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["status", "product_type", "vendor", "collection_id", "since_id", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)

class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(...)

@mcp.tool(name="shopify_get_product", annotations={"readOnlyHint": True})
async def shopify_get_product(params: GetProductInput) -> str:
    """Retrieve a single product by ID, including all variants and images."""
    try:
        data = await _request("GET", f"products/{params.product_id}.json")
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class CreateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:        str                            = Field(..., min_length=1)
    body_html:    Optional[str]                  = Field(default=None)
    vendor:       Optional[str]                  = Field(default=None)
    product_type: Optional[str]                  = Field(default=None)
    tags:         Optional[str]                  = Field(default=None)
    status:       Optional[str]                  = Field(default="draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None)
    options:      Optional[List[Dict[str, Any]]] = Field(default=None)
    images:       Optional[List[Dict[str, Any]]] = Field(default=None)

@mcp.tool(name="shopify_create_product", annotations={"readOnlyHint": False})
async def shopify_create_product(params: CreateProductInput) -> str:
    """Create a new product in the Shopify store."""
    try:
        product: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "vendor", "product_type", "tags", "status", "variants", "options", "images"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("POST", "products.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class UpdateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:   int                            = Field(...)
    title:        Optional[str]                  = Field(default=None)
    body_html:    Optional[str]                  = Field(default=None)
    vendor:       Optional[str]                  = Field(default=None)
    product_type: Optional[str]                  = Field(default=None)
    tags:         Optional[str]                  = Field(default=None)
    status:       Optional[str]                  = Field(default=None)
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None)

@mcp.tool(name="shopify_update_product", annotations={"readOnlyHint": False})
async def shopify_update_product(params: UpdateProductInput) -> str:
    """Update an existing product. Only provided fields are changed."""
    try:
        product: Dict[str, Any] = {}
        for field in ["title", "body_html", "vendor", "product_type", "tags", "status", "variants"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("PUT", f"products/{params.product_id}.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class DeleteProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(...)

@mcp.tool(name="shopify_delete_product", annotations={"readOnlyHint": False, "destructiveHint": True})
async def shopify_delete_product(params: DeleteProductInput) -> str:
    """Permanently delete a product. This cannot be undone."""
    try:
        await _request("DELETE", f"products/{params.product_id}.json")
        return f"Product {params.product_id} deleted."
    except Exception as e:
        return _error(e)

class ProductCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:       Optional[str] = Field(default=None)
    vendor:       Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_count_products", annotations={"readOnlyHint": True})
async def shopify_count_products(params: ProductCountInput) -> str:
    """Get the total count of products, optionally filtered."""
    try:
        p: Dict[str, Any] = {}
        for field in ["status", "vendor", "product_type"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════════

class ListOrdersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:               Optional[int] = Field(default=50, ge=1, le=250)
    status:              Optional[str] = Field(default="any")
    financial_status:    Optional[str] = Field(default=None)
    fulfillment_status:  Optional[str] = Field(default=None)
    since_id:            Optional[int] = Field(default=None)
    created_at_min:      Optional[str] = Field(default=None)
    created_at_max:      Optional[str] = Field(default=None)
    fields:              Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_orders", annotations={"readOnlyHint": True})
async def shopify_list_orders(params: ListOrdersInput) -> str:
    """List orders with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit, "status": params.status}
        for field in ["financial_status", "fulfillment_status", "since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)

class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(...)

@mcp.tool(name="shopify_get_order", annotations={"readOnlyHint": True})
async def shopify_get_order(params: GetOrderInput) -> str:
    """Retrieve a single order by ID with full details."""
    try:
        data = await _request("GET", f"orders/{params.order_id}.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)

class OrderCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:             Optional[str] = Field(default="any")
    financial_status:   Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_count_orders", annotations={"readOnlyHint": True})
async def shopify_count_orders(params: OrderCountInput) -> str:
    """Get total order count, optionally filtered."""
    try:
        p: Dict[str, Any] = {"status": params.status}
        for field in ["financial_status", "fulfillment_status"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)

class CloseOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(...)

@mcp.tool(name="shopify_close_order", annotations={"readOnlyHint": False})
async def shopify_close_order(params: CloseOrderInput) -> str:
    """Close an order (marks it as completed)."""
    try:
        data = await _request("POST", f"orders/{params.order_id}/close.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)

class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int            = Field(...)
    reason:   Optional[str]  = Field(default=None)
    email:    Optional[bool] = Field(default=True)
    restock:  Optional[bool] = Field(default=False)

@mcp.tool(name="shopify_cancel_order", annotations={"readOnlyHint": False, "destructiveHint": True})
async def shopify_cancel_order(params: CancelOrderInput) -> str:
    """Cancel an order."""
    try:
        body: Dict[str, Any] = {}
        for field in ["reason", "email", "restock"]:
            val = getattr(params, field)
            if val is not None:
                body[field] = val
        data = await _request("POST", f"orders/{params.order_id}/cancel.json", body=body)
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════

class ListCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int] = Field(default=50, ge=1, le=250)
    since_id:       Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None)
    created_at_max: Optional[str] = Field(default=None)
    fields:         Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_customers", annotations={"readOnlyHint": True})
async def shopify_list_customers(params: ListCustomersInput) -> str:
    """List customers from the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for f in ["since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, f)
            if val is not None:
                p[f] = val
        data = await _request("GET", "customers.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)

class SearchCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str           = Field(..., min_length=1)
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_search_customers", annotations={"readOnlyHint": True})
async def shopify_search_customers(params: SearchCustomersInput) -> str:
    """Search customers by name, email, or other fields."""
    try:
        p = {"query": params.query, "limit": params.limit}
        data = await _request("GET", "customers/search.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)

class GetCustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(...)

@mcp.tool(name="shopify_get_customer", annotations={"readOnlyHint": True})
async def shopify_get_customer(params: GetCustomerInput) -> str:
    """Retrieve a single customer by ID."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}.json")
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class CreateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    first_name:        Optional[str]                  = Field(default=None)
    last_name:         Optional[str]                  = Field(default=None)
    email:             Optional[str]                  = Field(default=None)
    phone:             Optional[str]                  = Field(default=None)
    tags:              Optional[str]                  = Field(default=None)
    note:              Optional[str]                  = Field(default=None)
    addresses:         Optional[List[Dict[str, Any]]] = Field(default=None)
    send_email_invite: Optional[bool]                 = Field(default=False)

@mcp.tool(name="shopify_create_customer", annotations={"readOnlyHint": False})
async def shopify_create_customer(params: CreateCustomerInput) -> str:
    """Create a new customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note", "addresses", "send_email_invite"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("POST", "customers.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class UpdateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    customer_id: int           = Field(...)
    first_name:  Optional[str] = Field(default=None)
    last_name:   Optional[str] = Field(default=None)
    email:       Optional[str] = Field(default=None)
    phone:       Optional[str] = Field(default=None)
    tags:        Optional[str] = Field(default=None)
    note:        Optional[str] = Field(default=None)

@mcp.tool(name="shopify_update_customer", annotations={"readOnlyHint": False})
async def shopify_update_customer(params: UpdateCustomerInput) -> str:
    """Update an existing customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("PUT", f"customers/{params.customer_id}.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class CustomerOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int           = Field(...)
    limit:       Optional[int] = Field(default=50, ge=1, le=250)
    status:      Optional[str] = Field(default="any")

@mcp.tool(name="shopify_get_customer_orders", annotations={"readOnlyHint": True})
async def shopify_get_customer_orders(params: CustomerOrdersInput) -> str:
    """Get all orders for a specific customer."""
    try:
        p = {"limit": params.limit, "status": params.status}
        data = await _request("GET", f"customers/{params.customer_id}/orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════

class ListCollectionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:           Optional[int] = Field(default=50, ge=1, le=250)
    since_id:        Optional[int] = Field(default=None)
    collection_type: Optional[str] = Field(default="custom")

@mcp.tool(name="shopify_list_collections", annotations={"readOnlyHint": True})
async def shopify_list_collections(params: ListCollectionsInput) -> str:
    """List custom or smart collections."""
    try:
        endpoint = "custom_collections.json" if params.collection_type == "custom" else "smart_collections.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        data = await _request("GET", endpoint, params=p)
        key  = "custom_collections" if params.collection_type == "custom" else "smart_collections"
        collections = data.get(key, [])
        return _fmt({"count": len(collections), "collections": collections})
    except Exception as e:
        return _error(e)

class GetCollectionProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int           = Field(...)
    limit:         Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_get_collection_products", annotations={"readOnlyHint": True})
async def shopify_get_collection_products(params: GetCollectionProductsInput) -> str:
    """Get all products in a specific collection."""
    try:
        p = {"limit": params.limit, "collection_id": params.collection_id}
        data = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

class ListInventoryLocationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_list_locations", annotations={"readOnlyHint": True})
async def shopify_list_locations(params: ListInventoryLocationsInput) -> str:
    """List all inventory locations for the store."""
    try:
        data = await _request("GET", "locations.json")
        locations = data.get("locations", [])
        return _fmt({"count": len(locations), "locations": locations})
    except Exception as e:
        return _error(e)

class GetInventoryLevelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location_id:        Optional[int] = Field(default=None)
    inventory_item_ids: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_get_inventory_levels", annotations={"readOnlyHint": True})
async def shopify_get_inventory_levels(params: GetInventoryLevelsInput) -> str:
    """Get inventory levels for specific locations or inventory items."""
    try:
        p: Dict[str, Any] = {}
        if params.location_id:
            p["location_ids"] = params.location_id
        if params.inventory_item_ids:
            p["inventory_item_ids"] = params.inventory_item_ids
        data = await _request("GET", "inventory_levels.json", params=p)
        levels = data.get("inventory_levels", [])
        return _fmt({"count": len(levels), "inventory_levels": levels})
    except Exception as e:
        return _error(e)

class SetInventoryLevelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory_item_id: int = Field(...)
    location_id:       int = Field(...)
    available:         int = Field(...)

@mcp.tool(name="shopify_set_inventory_level", annotations={"readOnlyHint": False})
async def shopify_set_inventory_level(params: SetInventoryLevelInput) -> str:
    """Set the available inventory for an item at a location."""
    try:
        body = {"inventory_item_id": params.inventory_item_id, "location_id": params.location_id, "available": params.available}
        data = await _request("POST", "inventory_levels/set.json", body=body)
        return _fmt(data.get("inventory_level", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FULFILLMENTS
# ═══════════════════════════════════════════════════════════════════════════

class ListFulfillmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int           = Field(...)
    limit:    Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_fulfillments", annotations={"readOnlyHint": True})
async def shopify_list_fulfillments(params: ListFulfillmentsInput) -> str:
    """List fulfillments for a specific order."""
    try:
        p = {"limit": params.limit}
        data = await _request("GET", f"orders/{params.order_id}/fulfillments.json", params=p)
        fulfillments = data.get("fulfillments", [])
        return _fmt({"count": len(fulfillments), "fulfillments": fulfillments})
    except Exception as e:
        return _error(e)

class CreateFulfillmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id:         int                            = Field(...)
    location_id:      int                            = Field(...)
    tracking_number:  Optional[str]                  = Field(default=None)
    tracking_company: Optional[str]                  = Field(default=None)
    tracking_url:     Optional[str]                  = Field(default=None)
    line_items:       Optional[List[Dict[str, Any]]] = Field(default=None)
    notify_customer:  Optional[bool]                 = Field(default=True)

@mcp.tool(name="shopify_create_fulfillment", annotations={"readOnlyHint": False})
async def shopify_create_fulfillment(params: CreateFulfillmentInput) -> str:
    """Create a fulfillment for an order (ship items)."""
    try:
        fulfillment: Dict[str, Any] = {"location_id": params.location_id}
        for field in ["tracking_number", "tracking_company", "tracking_url", "line_items", "notify_customer"]:
            val = getattr(params, field)
            if val is not None:
                fulfillment[field] = val
        data = await _request("POST", f"orders/{params.order_id}/fulfillments.json", body={"fulfillment": fulfillment})
        return _fmt(data.get("fulfillment", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SHOP INFO
# ═══════════════════════════════════════════════════════════════════════════

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_get_shop", annotations={"readOnlyHint": True})
async def shopify_get_shop(params: EmptyInput) -> str:
    """Get store information: name, domain, plan, currency, timezone, etc."""
    try:
        data = await _request("GET", "shop.json")
        return _fmt(data.get("shop", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════

class ListWebhooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    topic: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_webhooks", annotations={"readOnlyHint": True})
async def shopify_list_webhooks(params: ListWebhooksInput) -> str:
    """List configured webhooks."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.topic:
            p["topic"] = params.topic
        data = await _request("GET", "webhooks.json", params=p)
        webhooks = data.get("webhooks", [])
        return _fmt({"count": len(webhooks), "webhooks": webhooks})
    except Exception as e:
        return _error(e)

class CreateWebhookInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic:   str           = Field(...)
    address: str           = Field(...)
    format:  Optional[str] = Field(default="json")

@mcp.tool(name="shopify_create_webhook", annotations={"readOnlyHint": False})
async def shopify_create_webhook(params: CreateWebhookInput) -> str:
    """Create a new webhook subscription."""
    try:
        webhook = {"topic": params.topic, "address": params.address, "format": params.format}
        data = await _request("POST", "webhooks.json", body={"webhook": webhook})
        return _fmt(data.get("webhook", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# THEMES
# ═══════════════════════════════════════════════════════════════════════════

class ListThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_list_themes", annotations={"readOnlyHint": True})
async def shopify_list_themes(params: ListThemesInput) -> str:
    """List all themes in the store. Shows which theme is active."""
    try:
        data = await _request("GET", "themes.json")
        themes = data.get("themes", [])
        return _fmt({"count": len(themes), "themes": themes})
    except Exception as e:
        return _error(e)


class ListThemeFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID (get from shopify_list_themes)")

@mcp.tool(name="shopify_list_theme_files", annotations={"readOnlyHint": True})
async def shopify_list_theme_files(params: ListThemeFilesInput) -> str:
    """List all files in a theme (sections, templates, snippets, assets, etc.)."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json")
        assets = data.get("assets", [])
        return _fmt({"count": len(assets), "files": [a.get("key") for a in assets]})
    except Exception as e:
        return _error(e)


class GetThemeFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    file_key: str = Field(..., description="File path e.g. 'sections/header.liquid' or 'assets/base.css'")

@mcp.tool(name="shopify_get_theme_file", annotations={"readOnlyHint": True})
async def shopify_get_theme_file(params: GetThemeFileInput) -> str:
    """Read the contents of a theme file."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json", params={"asset[key]": params.file_key})
        asset = data.get("asset", data)
        return _fmt(asset)
    except Exception as e:
        return _error(e)


class UpdateThemeFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    file_key: str = Field(..., description="File path e.g. 'sections/header.liquid'")
    value:    str = Field(..., description="Full file contents to write")

@mcp.tool(name="shopify_update_theme_file", annotations={"readOnlyHint": False})
async def shopify_update_theme_file(params: UpdateThemeFileInput) -> str:
    """Write/update a theme file. Replaces the entire file content."""
    try:
        body = {"asset": {"key": params.file_key, "value": params.value}}
        data = await _request("PUT", f"themes/{params.theme_id}/assets.json", body=body)
        asset = data.get("asset", data)
        return _fmt({"success": True, "key": asset.get("key"), "updated_at": asset.get("updated_at")})
    except Exception as e:
        return _error(e)


class DeleteThemeFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    file_key: str = Field(..., description="File path to delete")

@mcp.tool(name="shopify_delete_theme_file", annotations={"readOnlyHint": False, "destructiveHint": True})
async def shopify_delete_theme_file(params: DeleteThemeFileInput) -> str:
    """Delete a file from a theme. Cannot be undone."""
    try:
        await _request("DELETE", f"themes/{params.theme_id}/assets.json", params={"asset[key]": params.file_key})
        return f"File '{params.file_key}' deleted from theme {params.theme_id}."
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
