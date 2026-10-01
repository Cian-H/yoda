import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    from yoda.architecture.schema import DecisionPayload, QueryContext

    mo.md("# Yoda: System 1 Decision Engine Exploration")
    return DecisionPayload, QueryContext, mo


@app.cell
def __(DecisionPayload, QueryContext, mo):
    sample_payload = DecisionPayload(
        query="sample_decision",
        context=QueryContext(
            semantic_embedding=[0.25, -0.15, 0.78],
            symbolic_state={"rule_active": True},
        ),
        constraints=["require_valid_path"],
    )
    mo.ui.table([sample_payload.model_dump()])
    return (sample_payload,)


if __name__ == "__main__":
    app.run()
