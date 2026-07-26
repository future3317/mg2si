import pandas as pd


def recommendation_cards(frame: pd.DataFrame) -> list[dict]:
    fields = ["candidate_id", "workflow_branch", "recommendation_role", "model_path", "probability_efficacy", "probability_safety", "out_of_domain"]
    return [{field: row.get(field) for field in fields} for _, row in frame.iterrows()]

