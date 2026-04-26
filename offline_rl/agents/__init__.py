from agents.fql_agent import FQLAgent
from agents.iql_agent import IQLAgent
from agents.sacbc_agent import SACBCAgent
from agents.asbs_agent import ASBSAgent

agents = {
    "fql": FQLAgent,
    "iql": IQLAgent,
    "sacbc": SACBCAgent,
    "asbs": ASBSAgent,
}