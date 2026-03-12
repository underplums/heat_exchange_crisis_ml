from src.screening.run_information_score import InformationScorer

scorer = InformationScorer()

scenario_scores = []

for scenario in scenarios:

    score = scorer.compute_priority_score(
        probabilities=scenario["probabilities"],
        distance_to_train=scenario["distance"],
        predicted_class=scenario["predicted_class"]
    )

    scenario["priority_score"] = score["priority_score"]

    scenario_scores.append(scenario)


ranked = scorer.rank_scenarios(scenario_scores)

top_k = ranked[:10]