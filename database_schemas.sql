-- E-commerce Database Schemas
-- Generated automatically from CSV files

-- Table: p__l_march_2021
CREATE TABLE IF NOT EXISTS p__l_march_2021 (
    sku TEXT,
    style_id TEXT,
    catalog TEXT,
    category TEXT,
    weight TEXT,
    tp_1 TEXT,
    tp_2 TEXT,
    mrp_old TEXT,
    final_mrp_old TEXT,
    ajio_mrp TEXT,
    amazon_mrp TEXT,
    amazon_fba_mrp TEXT,
    flipkart_mrp TEXT,
    limeroad_mrp TEXT,
    myntra_mrp TEXT,
    paytm_mrp TEXT,
    snapdeal_mrp TEXT
);

-- Table: international_sale_report
CREATE TABLE IF NOT EXISTS international_sale_report (
    date TEXT,
    months TEXT,
    customer TEXT,
    style TEXT,
    sku TEXT,
    size TEXT,
    pcs TEXT,
    rate TEXT,
    gross_amt TEXT
);

-- Table: cloud_warehouse_compersion_chart
CREATE TABLE IF NOT EXISTS cloud_warehouse_compersion_chart (
    shiprocket TEXT,
    unnamed_1 TEXT,
    increff TEXT
);

-- Table: sale_report
CREATE TABLE IF NOT EXISTS sale_report (
    sku_code TEXT,
    design_no_ TEXT,
    stock TEXT,
    category TEXT,
    size TEXT,
    color TEXT
);

-- Table: expense_iigf
CREATE TABLE IF NOT EXISTS expense_iigf (
    recived_amount TEXT,
    unnamed_1 TEXT,
    expance TEXT,
    unnamed_3 TEXT
);

-- Table: amazon_sale_report
CREATE TABLE IF NOT EXISTS amazon_sale_report (
    order_id TEXT,
    date TEXT,
    status TEXT,
    fulfilment TEXT,
    sales_channel TEXT,
    ship_service_level TEXT,
    style TEXT,
    sku TEXT,
    category TEXT,
    size TEXT,
    asin TEXT,
    courier_status TEXT,
    qty INTEGER,
    currency TEXT,
    amount TEXT,
    ship_city TEXT,
    ship_state TEXT,
    ship_postal_code TEXT,
    ship_country TEXT,
    promotion_ids TEXT,
    b2b BOOLEAN,
    fulfilled_by TEXT,
    unnamed_22 TEXT
);

-- Table: may_2022
CREATE TABLE IF NOT EXISTS may_2022 (
    sku TEXT,
    style_id TEXT,
    catalog TEXT,
    category TEXT,
    weight TEXT,
    tp TEXT,
    mrp_old TEXT,
    final_mrp_old TEXT,
    ajio_mrp TEXT,
    amazon_mrp TEXT,
    amazon_fba_mrp TEXT,
    flipkart_mrp TEXT,
    limeroad_mrp TEXT,
    myntra_mrp TEXT,
    paytm_mrp TEXT,
    snapdeal_mrp TEXT
);

