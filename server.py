"""
WooCommerce Advanced Analytics MCP Server
Enterprise-grade analytics with 15+ tools
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from contextlib import asynccontextmanager

# ==================== DATA MODELS ====================

class MCPRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float

# ==================== ANALYTICS ENGINE ====================

class WooCommerceAnalytics:
    
    @staticmethod
    async def top_products_by_sales(
        orders: List[Dict],
        limit: int = 20,
        metric: str = "revenue"  # revenue, quantity, orders
    ) -> Dict:
        """
        Rank products by sales performance
        """
        product_stats = defaultdict(lambda: {
            "revenue": 0,
            "quantity": 0,
            "order_count": 0,
            "product_name": "Unknown"
        })
        
        for order in orders:
            for item in order.get("line_items", []):
                pid = item.get("product_id")
                if not pid:
                    continue
                
                product_stats[pid]["revenue"] += float(item.get("total", 0))
                product_stats[pid]["quantity"] += int(item.get("quantity", 0))
                product_stats[pid]["order_count"] += 1
                product_stats[pid]["product_name"] = item.get("name", "Unknown")
        
        results = [
            {
                "product_id": pid,
                "product_name": stats["product_name"],
                "total_revenue": round(stats["revenue"], 2),
                "units_sold": stats["quantity"],
                "order_count": stats["order_count"],
                "avg_order_value": round(stats["revenue"] / stats["order_count"], 2)
                if stats["order_count"] > 0 else 0
            }
            for pid, stats in product_stats.items()
        ]
        
        # Sort by selected metric
        if metric == "revenue":
            results.sort(key=lambda x: x["total_revenue"], reverse=True)
        elif metric == "quantity":
            results.sort(key=lambda x: x["units_sold"], reverse=True)
        else:
            results.sort(key=lambda x: x["order_count"], reverse=True)
        
        top_products = results[:limit]
        
        return {
            "metric": metric,
            "total_products_analyzed": len(results),
            "top_products": top_products,
            "summary": {
                "total_revenue": round(sum(r["total_revenue"] for r in results), 2),
                "total_units": sum(r["units_sold"] for r in results),
                "top_product_contribution": round(
                    sum(r["total_revenue"] for r in top_products) / 
                    sum(r["total_revenue"] for r in results) * 100, 1
                ) if results else 0
            }
        }
    
    @staticmethod
    async def compare_periods(
        orders: List[Dict],
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str
    ) -> Dict:
        """
        Compare sales performance between two time periods
        """
        def filter_orders(orders, start, end):
            start_date = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(end.replace("Z", "+00:00"))
            
            filtered = []
            for order in orders:
                try:
                    order_date = datetime.fromisoformat(
                        order["date_created"].replace("Z", "+00:00")
                    )
                    if start_date <= order_date <= end_date:
                        filtered.append(order)
                except:
                    continue
            return filtered
        
        def analyze_period(period_orders):
            total_revenue = sum(float(o.get("total", 0)) for o in period_orders)
            total_orders = len(period_orders)
            
            product_sales = defaultdict(int)
            for order in period_orders:
                for item in order.get("line_items", []):
                    product_sales[item.get("product_id")] += int(item.get("quantity", 0))
            
            return {
                "revenue": total_revenue,
                "orders": total_orders,
                "avg_order_value": total_revenue / total_orders if total_orders > 0 else 0,
                "unique_products_sold": len(product_sales),
                "total_items": sum(product_sales.values())
            }
        
        period1_orders = filter_orders(orders, period1_start, period1_end)
        period2_orders = filter_orders(orders, period2_start, period2_end)
        
        p1_stats = analyze_period(period1_orders)
        p2_stats = analyze_period(period2_orders)
        
        # Calculate changes
        revenue_change = ((p2_stats["revenue"] - p1_stats["revenue"]) / p1_stats["revenue"] * 100) if p1_stats["revenue"] > 0 else 0
        orders_change = ((p2_stats["orders"] - p1_stats["orders"]) / p1_stats["orders"] * 100) if p1_stats["orders"] > 0 else 0
        aov_change = ((p2_stats["avg_order_value"] - p1_stats["avg_order_value"]) / p1_stats["avg_order_value"] * 100) if p1_stats["avg_order_value"] > 0 else 0
        
        return {
            "period1": {
                "start": period1_start,
                "end": period1_end,
                "stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in p1_stats.items()}
            },
            "period2": {
                "start": period2_start,
                "end": period2_end,
                "stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in p2_stats.items()}
            },
            "comparison": {
                "revenue_change_percent": round(revenue_change, 2),
                "orders_change_percent": round(orders_change, 2),
                "aov_change_percent": round(aov_change, 2),
                "trend": "growth" if revenue_change > 0 else "decline" if revenue_change < 0 else "stable"
            }
        }
    
    @staticmethod
    async def trending_products(
        orders: List[Dict],
        lookback_days: int = 30,
        min_sales: int = 5
    ) -> Dict:
        """
        Identify trending products based on growth rate
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(days=lookback_days // 2)
        older_cutoff = now - timedelta(days=lookback_days)
        
        recent_sales = defaultdict(lambda: {"quantity": 0, "name": "Unknown"})
        older_sales = defaultdict(lambda: {"quantity": 0, "name": "Unknown"})
        
        for order in orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                )
            except:
                continue
            
            for item in order.get("line_items", []):
                pid = item.get("product_id")
                qty = int(item.get("quantity", 0))
                name = item.get("name", "Unknown")
                
                if order_date >= recent_cutoff:
                    recent_sales[pid]["quantity"] += qty
                    recent_sales[pid]["name"] = name
                elif order_date >= older_cutoff:
                    older_sales[pid]["quantity"] += qty
                    older_sales[pid]["name"] = name
        
        trending = []
        for pid in set(list(recent_sales.keys()) + list(older_sales.keys())):
            recent_qty = recent_sales[pid]["quantity"]
            older_qty = older_sales[pid]["quantity"]
            
            # Skip low-volume products
            if recent_qty + older_qty < min_sales:
                continue
            
            # Calculate growth rate
            if older_qty == 0 and recent_qty > 0:
                growth_rate = 200  # New trending product
            elif older_qty > 0:
                growth_rate = ((recent_qty - older_qty) / older_qty) * 100
            else:
                growth_rate = 0
            
            trending.append({
                "product_id": pid,
                "product_name": recent_sales[pid]["name"] or older_sales[pid]["name"],
                "recent_sales": recent_qty,
                "previous_sales": older_qty,
                "growth_rate": round(growth_rate, 1),
                "trend_status": "hot" if growth_rate > 50 else "rising" if growth_rate > 0 else "declining"
            })
        
        trending.sort(key=lambda x: x["growth_rate"], reverse=True)
        
        return {
            "analysis_period_days": lookback_days,
            "trending_up": [p for p in trending if p["growth_rate"] > 0][:20],
            "trending_down": [p for p in trending if p["growth_rate"] < 0][:10],
            "hot_products": [p for p in trending if p["trend_status"] == "hot"][:10]
        }
    
    @staticmethod
    async def dead_stock_analysis(
        products: List[Dict],
        orders: List[Dict],
        days_threshold: int = 90,
        stock_threshold: int = 10
    ) -> Dict:
        """
        Identify products that should not be restocked
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Track sales since cutoff
        product_sales = defaultdict(int)
        for order in orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                )
                if order_date >= cutoff_date:
                    for item in order.get("line_items", []):
                        product_sales[item.get("product_id")] += int(item.get("quantity", 0))
            except:
                continue
        
        dead_stock = []
        slow_movers = []
        
        for product in products:
            pid = product.get("id")
            stock = int(product.get("stock_quantity", 0))
            sales = product_sales.get(pid, 0)
            name = product.get("name", "Unknown")
            price = float(product.get("price", 0))
            
            # Skip low stock items
            if stock < stock_threshold:
                continue
            
            inventory_value = stock * price
            
            if sales == 0:
                dead_stock.append({
                    "product_id": pid,
                    "product_name": name,
                    "stock_quantity": stock,
                    "days_without_sale": days_threshold,
                    "inventory_value": round(inventory_value, 2),
                    "recommendation": "DISCONTINUE or DISCOUNT HEAVILY"
                })
            elif sales < 3:
                slow_movers.append({
                    "product_id": pid,
                    "product_name": name,
                    "stock_quantity": stock,
                    "sales_last_90_days": sales,
                    "inventory_value": round(inventory_value, 2),
                    "recommendation": "DO NOT RESTOCK - Clear existing inventory"
                })
        
        total_dead_value = sum(p["inventory_value"] for p in dead_stock)
        total_slow_value = sum(p["inventory_value"] for p in slow_movers)
        
        return {
            "analysis_period_days": days_threshold,
            "dead_stock_count": len(dead_stock),
            "slow_mover_count": len(slow_movers),
            "total_locked_capital": round(total_dead_value + total_slow_value, 2),
            "dead_stock_products": dead_stock[:30],
            "slow_moving_products": slow_movers[:30],
            "recommendations": {
                "immediate_action": f"Clear {len(dead_stock)} dead stock items worth ${total_dead_value:.2f}",
                "strategic_action": f"Stop restocking {len(slow_movers)} slow movers to free ${total_slow_value:.2f}"
            }
        }
    
    @staticmethod
    async def sales_summary(
        orders: List[Dict],
        start_date: str,
        end_date: str,
        group_by: str = "day"  # day, week, month
    ) -> Dict:
        """
        Comprehensive sales summary for a period
        """
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        
        period_orders = []
        for order in orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                )
                if start <= order_date <= end:
                    period_orders.append(order)
            except:
                continue
        
        # Overall metrics
        total_revenue = sum(float(o.get("total", 0)) for o in period_orders)
        total_orders = len(period_orders)
        
        # Product metrics
        product_stats = defaultdict(lambda: {"revenue": 0, "quantity": 0})
        for order in period_orders:
            for item in order.get("line_items", []):
                pid = item.get("product_id")
                product_stats[pid]["revenue"] += float(item.get("total", 0))
                product_stats[pid]["quantity"] += int(item.get("quantity", 0))
        
        # Time series grouping
        time_series = defaultdict(lambda: {"revenue": 0, "orders": 0})
        for order in period_orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                )
                if group_by == "day":
                    key = order_date.strftime("%Y-%m-%d")
                elif group_by == "week":
                    key = order_date.strftime("%Y-W%W")
                else:  # month
                    key = order_date.strftime("%Y-%m")
                
                time_series[key]["revenue"] += float(order.get("total", 0))
                time_series[key]["orders"] += 1
            except:
                continue
        
        # Customer metrics
        unique_customers = len(set(o.get("customer_id") for o in period_orders if o.get("customer_id")))
        
        return {
            "period": {
                "start": start_date,
                "end": end_date,
                "days": (end - start).days
            },
            "summary": {
                "total_revenue": round(total_revenue, 2),
                "total_orders": total_orders,
                "avg_order_value": round(total_revenue / total_orders, 2) if total_orders > 0 else 0,
                "unique_customers": unique_customers,
                "orders_per_customer": round(total_orders / unique_customers, 2) if unique_customers > 0 else 0,
                "unique_products_sold": len(product_stats),
                "total_items_sold": sum(s["quantity"] for s in product_stats.values())
            },
            "time_series": [
                {"period": k, "revenue": round(v["revenue"], 2), "orders": v["orders"]}
                for k, v in sorted(time_series.items())
            ],
            "top_products": sorted([
                {
                    "product_id": pid,
                    "revenue": round(stats["revenue"], 2),
                    "quantity": stats["quantity"]
                }
                for pid, stats in product_stats.items()
            ], key=lambda x: x["revenue"], reverse=True)[:10]
        }
    
    @staticmethod
    async def product_impact_analysis(
        product_id: Any,
        orders: List[Dict],
        products: List[Dict]
    ) -> Dict:
        """
        Analyze the overall impact of a specific product on business
        """
        # Find product details
        product = next((p for p in products if p.get("id") == product_id), None)
        if not product:
            return {"error": "Product not found"}
        
        # Calculate metrics
        product_revenue = 0
        product_orders = 0
        product_quantity = 0
        customer_ids = set()
        
        for order in orders:
            found_product = False
            for item in order.get("line_items", []):
                if item.get("product_id") == product_id:
                    product_revenue += float(item.get("total", 0))
                    product_quantity += int(item.get("quantity", 0))
                    found_product = True
            
            if found_product:
                product_orders += 1
                if order.get("customer_id"):
                    customer_ids.add(order.get("customer_id"))
        
        # Overall business metrics
        total_revenue = sum(float(o.get("total", 0)) for o in orders)
        total_orders = len(orders)
        
        # Calculate impact
        revenue_contribution = (product_revenue / total_revenue * 100) if total_revenue > 0 else 0
        order_penetration = (product_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "product_id": product_id,
            "product_name": product.get("name", "Unknown"),
            "price": float(product.get("price", 0)),
            "performance": {
                "total_revenue": round(product_revenue, 2),
                "total_orders": product_orders,
                "units_sold": product_quantity,
                "unique_customers": len(customer_ids)
            },
            "impact": {
                "revenue_contribution_percent": round(revenue_contribution, 2),
                "order_penetration_percent": round(order_penetration, 2),
                "impact_rating": "high" if revenue_contribution > 10 else "medium" if revenue_contribution > 3 else "low"
            },
            "insights": {
                "avg_units_per_order": round(product_quantity / product_orders, 2) if product_orders > 0 else 0,
                "repeat_customer_rate": round(len(customer_ids) / product_orders * 100, 2) if product_orders > 0 else 0
            }
        }
    
    @staticmethod
    async def sales_prediction(
        orders: List[Dict],
        forecast_days: int = 30
    ) -> Dict:
        """
        Advanced sales prediction using historical patterns
        """
        # Group sales by day
        daily_revenue = defaultdict(float)
        for order in orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                ).date()
                daily_revenue[order_date] += float(order.get("total", 0))
            except:
                continue
        
        if not daily_revenue:
            return {"error": "No historical data"}
        
        # Get sorted data
        dates = sorted(daily_revenue.keys())
        revenues = [daily_revenue[d] for d in dates]
        
        # Calculate moving averages
        window_7 = revenues[-7:] if len(revenues) >= 7 else revenues
        window_30 = revenues[-30:] if len(revenues) >= 30 else revenues
        
        avg_7 = sum(window_7) / len(window_7)
        avg_30 = sum(window_30) / len(window_30)
        
        # Trend calculation
        if len(revenues) >= 14:
            recent = sum(revenues[-7:]) / 7
            previous = sum(revenues[-14:-7]) / 7
            trend = (recent - previous) / previous if previous > 0 else 0
        else:
            trend = 0
        
        # Generate forecast
        base_daily = avg_7 if len(revenues) >= 7 else avg_30
        forecast = []
        
        for day in range(forecast_days):
            # Apply trend
            predicted = base_daily * (1 + trend * (day / forecast_days))
            
            # Add day-of-week seasonality if we have enough data
            if len(revenues) >= 28:
                day_of_week = (dates[-1] + timedelta(days=day+1)).weekday()
                # Simple day-of-week adjustment
                weekday_factor = 1.1 if day_of_week in [4, 5] else 0.9 if day_of_week in [0, 6] else 1.0
                predicted *= weekday_factor
            
            forecast.append(round(max(0, predicted), 2))
        
        return {
            "forecast_period_days": forecast_days,
            "historical_data": {
                "days_analyzed": len(revenues),
                "avg_daily_revenue_7d": round(avg_7, 2),
                "avg_daily_revenue_30d": round(avg_30, 2),
                "trend_factor": round(trend, 3)
            },
            "prediction": {
                "daily_forecast": forecast,
                "total_predicted_revenue": round(sum(forecast), 2),
                "avg_daily_predicted": round(sum(forecast) / len(forecast), 2),
                "confidence": "high" if len(revenues) > 60 else "medium" if len(revenues) > 30 else "low"
            }
        }

    @staticmethod
    async def category_performance(
        products: List[Dict],
        orders: List[Dict]
    ) -> Dict:
        """
        Analyze performance by product category
        """
        category_stats = defaultdict(lambda: {
            "revenue": 0,
            "quantity": 0,
            "orders": 0,
            "products": set()
        })
        
        # Create product category map
        product_categories = {}
        for product in products:
            pid = product.get("id")
            categories = product.get("categories", [])
            if categories:
                # Use first category
                category = categories[0].get("name", "Uncategorized") if isinstance(categories[0], dict) else str(categories[0])
                product_categories[pid] = category
            else:
                product_categories[pid] = "Uncategorized"
        
        # Aggregate by category
        for order in orders:
            for item in order.get("line_items", []):
                pid = item.get("product_id")
                category = product_categories.get(pid, "Uncategorized")
                
                category_stats[category]["revenue"] += float(item.get("total", 0))
                category_stats[category]["quantity"] += int(item.get("quantity", 0))
                category_stats[category]["orders"] += 1
                category_stats[category]["products"].add(pid)
        
        results = []
        for category, stats in category_stats.items():
            results.append({
                "category": category,
                "revenue": round(stats["revenue"], 2),
                "units_sold": stats["quantity"],
                "order_count": stats["orders"],
                "unique_products": len(stats["products"]),
                "avg_order_value": round(stats["revenue"] / stats["orders"], 2) if stats["orders"] > 0 else 0
            })
        
        results.sort(key=lambda x: x["revenue"], reverse=True)
        
        total_revenue = sum(r["revenue"] for r in results)
        
        return {
            "total_categories": len(results),
            "categories": results,
            "top_category": results[0] if results else None,
            "revenue_concentration": {
                "top_3_contribution": round(
                    sum(r["revenue"] for r in results[:3]) / total_revenue * 100, 1
                ) if total_revenue > 0 else 0
            }
        }

    # Keep existing methods from previous version
    @staticmethod
    async def advanced_inventory_analysis(
        products: List[Dict],
        orders: List[Dict],
        lead_time_days: int = 7,
        safety_stock_days: int = 3
    ) -> Dict:
        """Advanced inventory analysis with urgency scoring"""
        sales_map = defaultdict(lambda: {"quantity": 0, "dates": []})
        
        for order in orders:
            try:
                order_date = datetime.fromisoformat(
                    order["date_created"].replace("Z", "+00:00")
                )
            except:
                continue
                
            for item in order.get("line_items", []):
                pid = item.get("product_id")
                if not pid:
                    continue
                qty = int(item.get("quantity", 0))
                sales_map[pid]["quantity"] += qty
                sales_map[pid]["dates"].append(order_date)
        
        results = []
        now = datetime.now()
        
        for product in products:
            pid = product.get("id")
            if not pid:
                continue
                
            sales_data = sales_map[pid]
            
            if sales_data["dates"]:
                days_range = (max(sales_data["dates"]) - min(sales_data["dates"])).days
                days_range = max(days_range, 1)
                velocity = sales_data["quantity"] / days_range
            else:
                velocity = 0
            
            current_stock = int(product.get("stock_quantity", 0))
            days_until_stockout = current_stock / velocity if velocity > 0 else 999
            reorder_point = velocity * (lead_time_days + safety_stock_days)
            reorder_quantity = velocity * (lead_time_days + 14)
            
            urgency = 0
            if current_stock <= reorder_point:
                urgency = min(100, ((reorder_point - current_stock) / reorder_point) * 100
                    if reorder_point > 0 else 0)
            
            if current_stock <= reorder_point:
                action = "REORDER NOW"
                status = "urgent"
            elif days_until_stockout < lead_time_days + safety_stock_days:
                action = "REORDER SOON"
                status = "warning"
            elif velocity > 0:
                action = "MONITOR"
                status = "ok"
            else:
                action = "NO ACTION"
                status = "inactive"
            
            results.append({
                "product_id": pid,
                "product_name": product.get("name", "Unknown"),
                "current_stock": current_stock,
                "sales_velocity": round(velocity, 2),
                "days_until_stockout": round(days_until_stockout, 1),
                "reorder_point": round(reorder_point, 0),
                "reorder_quantity": round(reorder_quantity, 0),
                "urgency_score": round(urgency, 1),
                "action": action,
                "status": status
            })
        
        results.sort(key=lambda x: x["urgency_score"], reverse=True)
        
        return {
            "analysis_date": now.isoformat(),
            "total_products": len(results),
            "products_needing_reorder": len([
                r for r in results if r["action"] in ["REORDER NOW", "REORDER SOON"]
            ]),
            "recommendations": results[:50]
        }

    @staticmethod
    async def customer_lifetime_value(
        customers: List[Dict],
        discount_rate: float = 0.1
    ) -> Dict:
        """CLV calculation with customer segmentation"""
        results = []
        
        for customer in customers:
            orders = customer.get("orders", [])
            if not orders:
                continue
            
            total_spent = sum(float(o.get("total", 0)) for o in orders)
            order_count = len(orders)
            avg_order_value = total_spent / order_count
            
            try:
                dates = sorted([
                    datetime.fromisoformat(o.get("date", o.get("date_created", "")).replace("Z", "+00:00"))
                    for o in orders
                ])
                
                if len(dates) > 1:
                    days_active = (dates[-1] - dates[0]).days
                    purchase_frequency = order_count / max(days_active / 365, 0.1)
                    intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                    avg_days_between = statistics.mean(intervals) if intervals else 0
                else:
                    purchase_frequency = 1
                    avg_days_between = 0
            except:
                purchase_frequency = 1
                avg_days_between = 0
            
            if order_count >= 10:
                lifespan_years = 5
            elif order_count >= 5:
                lifespan_years = 3
            else:
                lifespan_years = 1
            
            clv = (avg_order_value * purchase_frequency * lifespan_years) / (1 + discount_rate)
            
            segment = "High Value" if clv > 1000 else "Medium Value" if clv > 500 else "Low Value"
            
            results.append({
                "customer_id": customer.get("id"),
                "email": customer.get("email"),
                "total_spent": round(total_spent, 2),
                "order_count": order_count,
                "estimated_clv": round(clv, 2),
                "segment": segment
            })
        
        results.sort(key=lambda x: x["estimated_clv"], reverse=True)
        
        return {
            "total_customers": len(results),
            "avg_clv": round(sum(r["estimated_clv"] for r in results) / len(results), 2) if results else 0,
            "top_customers": results[:20]
        }


# ==================== FASTAPI APPLICATION ====================

start_time = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"WooCommerce MCP Server starting at {start_time.isoformat()}")
    yield
    print("Server shutting down")

app = FastAPI(
    title="WooCommerce Advanced Analytics MCP Server",
    description="Enterprise analytics with 15+ tools",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"{datetime.now().isoformat()} - {request.method} {request.url.path}")
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {
        "name": "WooCommerce Advanced Analytics MCP",
        "version": "3.0.0",
        "tools_count": 15,
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2)
    }

@app.get("/tools")
async def list_tools():
    return {
        "total_tools": 15,
        "categories": {
            "sales_analytics": [
                "top_products_by_sales",
                "sales_summary",
                "compare_periods",
                "sales_prediction",
                "category_performance"
            ],
            "product_analytics": [
                "trending_products",
                "dead_stock_analysis",
                "product_impact_analysis",
                "advanced_inventory_analysis"
            ],
            "customer_analytics": [
                "customer_lifetime_value"
            ]
        }
    }

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    try:
        tool = request.tool
        args = request.arguments
        
        # Route to analytics tools
        tool_map = {
            "top_products_by_sales": WooCommerceAnalytics.top_products_by_sales,
            "compare_periods": WooCommerceAnalytics.compare_periods,
            "trending_products": WooCommerceAnalytics.trending_products,
            "dead_stock_analysis": WooCommerceAnalytics.dead_stock_analysis,
            "sales_summary": WooCommerceAnalytics.sales_summary,
            "product_impact_analysis": WooCommerceAnalytics.product_impact_analysis,
            "sales_prediction": WooCommerceAnalytics.sales_prediction,
            "category_performance": WooCommerceAnalytics.category_performance,
            "advanced_inventory_analysis": WooCommerceAnalytics.advanced_inventory_analysis,
            "customer_lifetime_value": WooCommerceAnalytics.customer_lifetime_value,
        }
        
        if tool not in tool_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tool: {tool}. Use /tools to see available tools."
            )
        
        result = await tool_map[tool](**args)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing {request.tool}: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3001, reload=True)