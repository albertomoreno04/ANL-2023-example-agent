import json
import os
from pathlib import Path
import time

from utils.runners import run_tournament

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement.
tournament_settings = {
    "agents": [
        {
            "class": "agents.stupid_agent.stupid_agent.StupidAgent",
        },
        # {
        #     "class": "agents.group32_agent.group32_agent.Group32Agent",
        #     "parameters": {"storage_dir": "agent_storage/Group32Agent"},
        # },
        {
            "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
            "parameters": {"storage_dir": "agent_storage/DreamTeam109Agent"},
        },
    ],
    "profile_sets": [
        ["domains/domain01/profileA.json", "domains/domain01/profileB.json"],
        ["domains/domain02/profileA.json", "domains/domain02/profileB.json"],
        ["domains/domain03/profileA.json", "domains/domain03/profileB.json"],
        ["domains/domain04/profileA.json", "domains/domain04/profileB.json"],
        ["domains/domain05/profileA.json", "domains/domain05/profileB.json"],
        ["domains/domain06/profileA.json", "domains/domain06/profileB.json"],
        ["domains/domain07/profileA.json", "domains/domain07/profileB.json"],
        ["domains/domain08/profileA.json", "domains/domain08/profileB.json"],
        ["domains/domain09/profileA.json", "domains/domain09/profileB.json"],
        ["domains/domain10/profileA.json", "domains/domain10/profileB.json"],
        ["domains/domain11/profileA.json", "domains/domain11/profileB.json"],
        ["domains/domain12/profileA.json", "domains/domain12/profileB.json"],
        ["domains/domain13/profileA.json", "domains/domain13/profileB.json"],
        ["domains/domain14/profileA.json", "domains/domain14/profileB.json"],
        ["domains/domain15/profileA.json", "domains/domain15/profileB.json"],
        ["domains/domain16/profileA.json", "domains/domain16/profileB.json"],
        ["domains/domain17/profileA.json", "domains/domain17/profileB.json"],
        ["domains/domain18/profileA.json", "domains/domain18/profileB.json"],

    ],
    "deadline_time_ms": 10000,
}

# run a session and obtain results in dictionaries
tournament_steps, tournament_results, tournament_results_summary = run_tournament(tournament_settings)

# save the tournament settings for reference
with open(RESULTS_DIR.joinpath("tournament_steps.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_steps, indent=2))
# save the tournament results
with open(RESULTS_DIR.joinpath("tournament_results.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_results, indent=2))
# save the tournament results summary
tournament_results_summary.to_csv(RESULTS_DIR.joinpath("tournament_results_summary.csv"))
