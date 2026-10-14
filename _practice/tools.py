from langchain_core.tools import tool

weight_units = {
    "g" : 1, 
    "kg" : 1000,
    "pound" : 453.59,
    "ounce" : 28.35
}

@tool
def weight_converter(value: float, from_unit : str, to_unit: str) -> float:
    '''to convert weight "from_unit" to "to_unit"..'''
    
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()
    
    if from_unit not in weight_units:
        return f"Error: Unknown Unit: {from_unit}. Supported units: {list(weight_units.keys())}"
    if to_unit not in weight_units:
        return f"Error: Unknown Unit: {to_unit}. Supported units: {list(weight_units.keys())}"
    
    grams = value * weight_units[from_unit]
    return grams/weight_units[to_unit]

currency = {
    "INR" : 1,
    "USD" : 95.79,
    "YUAN" : 14.30,
    "DIRHAMS" : 26.08,
    "POUND" : 127.98,
    "EURO" : 110.13,
    "RIAL" : 249.37,
}

@tool
def currency_converter(value: float, from_currency : str, to_currency: str) -> float:
    '''to convert currency "from_currency" to "to_currency"..'''
    
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()
    
    if from_currency not in currency:
        return f"Error: Unknown currency: {from_currency}. Supported currencies: {list(currency.keys())}"
    if to_currency not in currency:
        return f"Error: Unknown currency: {to_currency}. Supported currencies: {list(currency.keys())}"
    
    INR = value * currency[from_currency]
    return INR/currency[to_currency]

@tool
def budget_calculator(price: float, tax_rate: float, budget: float):
    '''to calculate tax/budget'''
    
    tax = (price * tax_rate)/100
    cost = price + tax
    
    report = {
        "price" : price,
        "tax" : tax,
        "total" : cost,
        "remaining" : budget - cost if cost <= budget else 0.0,
        "over by" : cost - budget if cost > budget else 0.0
    }
    p, t, c = report["price"], report["tax"], report["total"]
    
    lines = [
        f"Price: ₹{p:,.2f}",
        f"Tax ({tax_rate}%): ₹{t:,.2f}",
        f"total_cost_inr: ₹{c:,.2f}",
    ]
    
    if report["remaining"]:
        lines.append(f"✅ Fits! You'll have ₹{report['remaining']:,.2f} left.")
    else:
        lines.append(f"❌ Over budget by ₹{report['over by']:,.2f}.")
    
    result = "\n".join(lines)
    print(result)
    return result
    