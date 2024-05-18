import os

from dotenv import load_dotenv


def pytest_configure(config):
    load_dotenv(dotenv_path=".env")

    token_var = "SPECKLE_TOKEN"
    server_var = "SPECKLE_SERVER_URL"
    token = os.getenv(token_var)
    server = os.getenv(server_var)

    if not token:
        raise ValueError(f"Cannot run tests without a {token_var} environment variable")

    if not server:
        raise ValueError(
            f"Cannot run tests without a {server_var} environment variable"
        )

    # Set the token as an attribute on the config object
    config.SPECKLE_TOKEN = token
    config.SPECKLE_SERVER_URL = server

    project_var = "SPECKLE_PROJECT_ID"
    model_var = "SPECKLE_MODEL_ID"
    version_var = "SPECKLE_VERSION_ID"
    project_id = os.getenv(project_var)
    model_id = os.getenv(model_var)
    version_id = os.getenv(version_var)

    config.PROJECT_ID = project_id
    config.MODEL_ID = model_id
    config.VERSION_ID = version_id
    config.FUNCTION_ID = "123456"

    config.FUNCTION_NAME = "convert to GLTF"
    config.AUTOMATION_NAME = "local test"
    config.BRANCH_NAME = "main"
