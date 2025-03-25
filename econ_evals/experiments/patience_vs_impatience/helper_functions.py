def calc_future_value(interest_rate, timelength, value=100):
    return value * pow(interest_rate, timelength)


def calc_reliability_score(df, cutoff):
    df["reward_diff"] = df.apply(
        lambda row: (
            row["future_reward"]
            if row["value"] > cutoff
            else (1 - row["future_reward"])
            if row["value"] < cutoff
            else None
        ),
        axis=1,
    )

    # Drop rows where reward_diff is None
    result_df = df.dropna(subset=["reward_diff"])

    average_diff = result_df["reward_diff"].mean()
    return average_diff


def detect_horizon_str(dirname: str) -> str:
    if "one_month" in dirname:
        return "one_month"
    elif "six_months" in dirname:
        return "six_months"
    elif "one_year" in dirname:
        return "one_year"
    elif "five_years" in dirname:
        return "five_years"
    else:
        raise NotImplementedError


def horizon_str_to_horizon_length(horizon_str: str) -> float:
    if horizon_str == "one_month":
        return 1 / 12
    elif horizon_str == "six_months":
        return 1 / 2
    elif horizon_str == "one_year":
        return 1
    elif horizon_str == "five_years":
        return 5
    else:
        raise NotImplementedError
