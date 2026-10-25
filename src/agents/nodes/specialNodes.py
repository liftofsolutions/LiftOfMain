"""The two special nodes every agent's graph has: where a run comes in,
and where it goes out.

    graph.add_node(INPUT_NODE_ID, InputNode())
    graph.add_node(OUTPUT_NODE_ID, OutputNode())

Edges name them by their ids, "input" and "output", as they name every
other node by its id. Neither is reserved from the node manager: an
agent builds both, and they hold nothing - no credential, no config, no
model.

* `InputNode` starts the conversation in `messages` with the message the
  run was given - by a trigger, or a query - the state's `inputMessage`.
* `OutputNode` takes the run's `answer`, as `answerOf` gives it: the
  last reply of a node that answers - a chat model node's, as it keeps
  it in the state's `answer` - so a node that only picks where the run
  goes next, a reviewer passing a reply, is not the one heard. A run in
  which none answered goes out with the last message: a guard turning a
  message away, say.

A node picking where a run goes next reads the output node as
OUTPUT_ABOUT: the end of the run, its own answer going out as the
agent's.
"""

from langchain_core.messages import HumanMessage

# The ids edges name the two nodes by: where a run comes in, and where it
# goes out.
INPUT_NODE_ID = "input"
OUTPUT_NODE_ID = "output"
SPECIAL_NODE_IDS = (INPUT_NODE_ID, OUTPUT_NODE_ID)

# What a node picking where a run goes next reads of the output node.
OUTPUT_ABOUT = "the end of the run: your answer goes out as the agent's."


class InputNode:
    """Where a run comes in: it starts the conversation with the message
    the run was given."""

    nodeId = INPUT_NODE_ID

    async def __call__(self, state: dict) -> dict:
        """The conversation, begun with the message."""
        return {"messages": [HumanMessage(state["inputMessage"])]}


def answerOf(state: dict) -> str:
    """The run's answer: the last reply of a node that answers, else the
    last message's text."""
    return state.get("answer") or state["messages"][-1].text


class OutputNode:
    """Where a run goes out, with its answer."""

    nodeId = OUTPUT_NODE_ID

    async def __call__(self, state: dict) -> dict:
        """The answer, as `answerOf` gives it."""
        return {"answer": answerOf(state)}
