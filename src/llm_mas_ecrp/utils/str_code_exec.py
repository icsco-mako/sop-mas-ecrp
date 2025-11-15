from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def run_str_code(
    problem: Dict,
    code: str,
) -> Union[Dict, str]:
    """
    Executes the given code as a string and calls the optimize function with the given problem's input data.
    If execution fails, returns an error message instead of raising an exception.

    Args:
        problem: A dictionary containing the problem data.
        code: A string of Python code to execute.

    Returns:
        A dictionary containing the result of the optimization, or a string error message if execution fails.
    """
    try:
        # Execute the provided code
        exec(code, globals())

        # Extract input data from the problem
        data = problem["sample"][0]["input"]

        # Call the optimize function with the input data
        result: Dict = optimize(data)

        return result

    except Exception as e:
        # Log the error and return an error message
        error_msg = f"Code execution failed: {e}"
        logger.error(error_msg)
        return {"obj_value": None, "error_msg": error_msg}
