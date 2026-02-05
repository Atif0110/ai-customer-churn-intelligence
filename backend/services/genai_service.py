from backend.llm.factory import get_llm

llm = get_llm()


def generate_explanation(ds_output):

    drivers = ", ".join(ds_output["drivers"])

    prompt = f"""
You are a senior customer retention strategist.

A churn model predicted:

Churn Probability: {ds_output['churn_probability']:.2f}
Risk Level: {ds_output['risk_level']}

Main Risk Drivers:
{drivers}

Explain:

1) Why this customer might churn
2) Two retention actions
3) Business impact if they churn

Be concise and business-friendly.
"""

    return llm.generate(prompt)
