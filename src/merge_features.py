def merge_all(customer, churn, calls, usage):
    df = customer.merge(churn, how="left", on="unique_customer_identifier")
    df = df.merge(calls, how="left", on="unique_customer_identifier")
    df = df.merge(usage, how="left", on="unique_customer_identifier")
    return df