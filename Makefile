.PHONY: test eval run langfuse-check

PORT ?= 8000

test:
	PYTHONPATH=src .venv/bin/pytest -q

eval:
	PYTHONPATH=src .venv/bin/python -m medscope.eval $(EVAL_ARGS)

# Workbench. PORT=8081 make run to move it.
run:
	@echo "workbench → http://localhost:$(PORT)/workbench"
	PYTHONPATH=src .venv/bin/uvicorn medscope.app:app --reload --port $(PORT)

langfuse-check:
	@echo "not implemented yet"
