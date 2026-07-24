def make_router_node(distiller, input_key="input", output_key="intent"):
    """Return a LangGraph-compatible node. Duck-typed on dict state so this
    module never imports langgraph (keeps core dependency-free)."""
    def node(state):
        intent, source, conf = distiller.route(state[input_key])
        return {output_key: intent, "route_source": source,
                "route_confidence": conf}
    return node
