"""Write inspectable traces from the original mock harness; no model is called."""
from dataclasses import asdict
import json
from pathlib import Path
import platform

from run import Session, MockWorld, MockProvider, SimulatedCrash, run_loop, compact_context


def record():
    denied, world = Session('denied'), MockWorld()
    denied_id = denied.request('increment', 10)
    denial = denied.dispatch(denied_id, world, set())
    denial_trace = dict(status=denial.status, counter=world.counter,
                        events=[asdict(event) for event in denied.events])

    budget, world = Session('budget'), MockWorld()
    status, receipts = run_loop(budget, MockProvider([('increment', 1)], repeat=True),
                                world, {'increment'}, 3)
    budget_trace = dict(status=status, counter=world.counter, tool_budget=3,
                        receipts=[asdict(receipt) for receipt in receipts],
                        events=[asdict(event) for event in budget.events])

    crash, world = Session('crash'), MockWorld()
    call_id = crash.request('increment', 7)
    try:
        crash.dispatch(call_id, world, {'increment'}, crash_after_effect=True)
    except SimulatedCrash:
        pass
    resumed = Session.restore(crash.snapshot())
    status, receipts = run_loop(resumed, MockProvider([('increment', 7)]), world, {'increment'}, 3)
    crash_trace = dict(status=status, counter=world.counter,
                       receipts=[asdict(receipt) for receipt in receipts],
                       events=[asdict(event) for event in resumed.events])

    rule = 'Do not increment the counter.'
    history = [rule, 'Result was 6.']
    summary, missing = compact_context(history, (rule,), lambda _: 'Result was 6.')
    return dict(python=platform.python_version(), scope='Original deterministic in-memory mock; no Claude or SDK execution',
                denial=denial_trace, budget=budget_trace, crash=crash_trace,
                compaction=dict(history=history, summary=summary, missing_constraints=missing))


if __name__ == '__main__':
    target = Path(__file__).with_name('results.json')
    target.write_text(json.dumps(record(), indent=2) + '\n', encoding='utf-8')
    print('Wrote results.json: denial, budget, unknown outcome, and compaction traces.')
