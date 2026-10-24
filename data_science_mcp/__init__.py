#!/usr/bin/env python

from agent_connector_sdk.connector import Connector

from data_science_mcp.auth import get_client
from data_science_mcp.mcp.mcp_data_engine import register_data_engine_tools
from data_science_mcp.mcp.mcp_deep_delegate import register_deep_delegate_tools
from data_science_mcp.mcp.mcp_quant import register_quant_tools
from data_science_mcp.mcp.mcp_trainers import register_trainer_tools
from data_science_mcp.mcp.mcp_training_data import register_training_data_tools
from data_science_mcp.mcp_server import (
    register_data_management_tools,
    register_interpretability_tools,
    register_model_evolution_tools,
    register_model_training_tools,
    register_prompts,
)
from data_science_mcp.ml_engine import MLEngine

connector = Connector(
    api=MLEngine,
    get_client=get_client,
    tool_registry=(
        ("model-training", "MODEL_TRAININGTOOL", register_model_training_tools),
        ("training-data", "MODEL_TRAININGTOOL", register_training_data_tools),
        ("trainer", "MODEL_TRAININGTOOL", register_trainer_tools),
        ("deep-delegate", "MODEL_TRAININGTOOL", register_deep_delegate_tools),
        ("model-evolution", "MODEL_EVOLUTIONTOOL", register_model_evolution_tools),
        ("interpretability", "INTERPRETABILITYTOOL", register_interpretability_tools),
        ("data-management", "DATA_MANAGEMENTTOOL", register_data_management_tools),
        ("data-engine", "DATA_ENGINETOOL", register_data_engine_tools),
    ),
    instructions="Data Science MCP Server for model training, evaluation, and evolution. Integrates with agent-connector-sdk IModelEvolver  for Agentic-iModels workflows.",
    action_tools=(("quant", "QUANTTOOL", register_quant_tools),),
    extra_registrations=(register_prompts,),
)
