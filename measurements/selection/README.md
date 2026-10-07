# Selection measurements

The reusable library blocks live in `jev_navigator.selection`: `scent`, `graph`, `rank`, `active` and `outline`. These scripts compose them with this evaluation's cases. No case vocabulary or provider route belongs in the library blocks.

`collect.py`, `evaluate.py`, `lab.py` and `native.py` retain the zero-cost ranking and exact stored-request replay. `prepare_trial.py` prepares compact outline Choices. `paid_prepare.py` freezes code-ranked queues and a single bounded LLM shortlist page per case without dispatching a request.

The paid runner has four explicit modes: `shape` captures the real role request without a provider, `guard` reviews the changed shapes, `planner` sends the frozen LLM shortlist pages and `judge` evaluates the admitted units through the native Engine entry point. Every physical send reserves projected dollars in the durable ledger before dispatch. Complete exact responses are reused on resume. An unresolved reservation stays charged against the cap. One runner process owns this ledger at a time.

The runtime uses the pinned Engine and JVN checkouts recorded in the report through `PYTHONPATH`. It needs the same parsing dependencies as that host, NumPy, and the TypeSafe SDK extra. Run `test_paid_native.py` explicitly in that host environment; it proves code and text admission, pending coverage, real role grouping and packet rendering. Library CI runs the durable reservation, exact-byte and typed shortlist tests in `tests/test_selection_paid.py`, including a real local HTTP provider when the SDK is installed.

`paid_support.py` owns money admission, resource checks and shortlist validation. `paid_client.py` owns exact HTTP requests and receipts. `paid_native.py` supplies ranked candidates to the pinned Engine pack builder and renders the three measured rooms from the same role observations. `paid_report.py` measures lab allocation, literal native delivery, judged reach, cost and service time. Source units are read one file at a time. Labels are used only after selection for measurement.

The authoritative report and raw paid receipts are under `~/.local/share/jvn-takeover/2026-10-03/search-design/case1/`. The paid cap and authorization are recorded there. Preparation is free; running the paid modes requires an explicitly authorized trial. Credentials come from the existing provider configuration and are never printed or saved in receipts.
