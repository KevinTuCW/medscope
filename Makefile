.PHONY: test eval run langfuse-check

test:
	PYTHONPATH=src .venv/bin/pytest -q

eval:
	PYTHONPATH=src .venv/bin/python -m medscope.eval $(EVAL_ARGS)

run:
	@echo "not implemented yet"

langfuse-check:
	@echo "not implemented yet"
