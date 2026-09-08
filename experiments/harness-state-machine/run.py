"""A deterministic, in-memory harness lab; not a production agent runtime."""

from dataclasses import dataclass, replace
import unittest


@dataclass(frozen=True)
class Event:
    sequence: int
    kind: str
    call_id: str
    tool: str = ""
    value: int | str | None = None


@dataclass(frozen=True)
class Snapshot:
    version: int
    session_id: str
    constraints: tuple[str, ...]
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Receipt:
    status: str
    value: int | str | None = None


class SimulatedCrash(Exception):
    pass


class MockWorld:
    """The only side effect is changing this object's integer counter."""

    supported_tools = frozenset({"double", "increment"})

    def __init__(self):
        self.counter = 0

    def execute(self, tool, value):
        if tool == "double":
            return value * 2
        if tool == "increment":
            self.counter += value
            return self.counter
        raise ValueError("unknown tool")


class Session:
    def __init__(self, session_id="demo", constraints=()):
        self.session_id = session_id
        self.constraints = tuple(constraints)
        self._events = ()

    @property
    def events(self):
        return self._events

    def append(self, kind, call_id, tool="", value=None):
        event = Event(len(self._events) + 1, kind, call_id, tool, value)
        self._events += (event,)

    def request(self, tool, value):
        if type(value) is not int:
            raise ValueError("this mock accepts only integer arguments")
        number = sum(e.kind == "requested" for e in self.events) + 1
        call_id = f"{self.session_id}:{number}"
        self.append("requested", call_id, tool, value)
        return call_id

    def dispatch(self, call_id, world, allowed_tools, crash_after_effect=False):
        history = [e for e in self.events if e.call_id == call_id]
        if not history or history[0].kind != "requested":
            raise ValueError("unknown call ID")
        # Reuse a terminal receipt before considering another execution.
        for event in history:
            if event.kind in {"denied", "result"}:
                return Receipt(event.kind, event.value)
        if any(e.kind == "started" for e in history):
            return Receipt("unknown", "started without a recorded result")
        request = history[0]
        allowed = request.tool in allowed_tools and request.tool in world.supported_tools
        self.append("permission_checked", call_id, value="allow" if allowed else "deny")
        if not allowed:
            self.append("denied", call_id, value="tool is not permitted")
            return Receipt("denied", "tool is not permitted")
        # The permission decision precedes both the start record and the effect.
        self.append("started", call_id)
        result = world.execute(request.tool, request.value)
        if crash_after_effect:
            raise SimulatedCrash("effect happened; result was not journaled")
        self.append("result", call_id, value=result)
        return Receipt("result", result)

    def snapshot(self):
        return Snapshot(1, self.session_id, self.constraints, self.events)

    def unresolved_calls(self):
        completed = {e.call_id for e in self.events if e.kind == "result"}
        return tuple(e.call_id for e in self.events
                     if e.kind == "started" and e.call_id not in completed)

    @classmethod
    def restore(cls, snapshot):
        if snapshot.version != 1:
            raise ValueError("unsupported snapshot version")
        session = cls(snapshot.session_id, snapshot.constraints)
        session._events = snapshot.events
        return session


class MockProvider:
    """Scripted proposals stand in for model output; no model/API is used."""

    def __init__(self, proposals, repeat=False):
        self.proposals = tuple(proposals)
        self.repeat = repeat

    def next(self, receipts):
        if not self.proposals:
            return None
        if self.repeat:
            return self.proposals[0]
        index = len(receipts)
        return self.proposals[index] if index < len(self.proposals) else None


def run_loop(session, provider, world, allowed_tools, tool_budget):
    if type(tool_budget) is not int or tool_budget < 0:
        raise ValueError("tool_budget must be a nonnegative integer")
    unknown = session.unresolved_calls()
    if unknown:
        return "unknown", tuple(session.dispatch(call_id, world, allowed_tools) for call_id in unknown)
    receipts = []
    for _ in range(tool_budget):
        proposal = provider.next(tuple(receipts))
        if proposal is None:
            return "finished", tuple(receipts)
        call_id = session.request(*proposal)
        receipts.append(session.dispatch(call_id, world, allowed_tools))
    return "budget_exhausted", tuple(receipts)


def compact_context(history, constraints, summarizer):
    """Exact-string retention check, deliberately weaker than semantic checking."""
    summary = summarizer(tuple(history))
    missing = tuple(rule for rule in constraints if rule not in summary)
    return summary, missing


class HarnessTests(unittest.TestCase):
    def test_provider_loop_and_permission_order(self):
        session, world = Session(), MockWorld()
        provider = MockProvider([("double", 3), ("increment", 2)])
        status, receipts = run_loop(session, provider, world, {"double", "increment"}, 3)
        self.assertEqual((status, receipts), ("finished", (Receipt("result", 6), Receipt("result", 2))))
        self.assertEqual(world.counter, 2)
        for call_id in ("demo:1", "demo:2"):
            self.assertEqual([e.kind for e in session.events if e.call_id == call_id],
                             ["requested", "permission_checked", "started", "result"])

    def test_denial_is_reused_even_if_policy_later_changes(self):
        session, world = Session(), MockWorld()
        call_id = session.request("increment", 10)
        first = session.dispatch(call_id, world, set())
        saved_events = session.events
        self.assertEqual(first.status, "denied")
        self.assertEqual(session.dispatch(call_id, world, {"increment"}), first)
        self.assertEqual((world.counter, session.events), (0, saved_events))

    def test_recorded_result_is_reused_after_restore(self):
        session, world = Session(), MockWorld()
        call_id = session.request("increment", 4)
        receipt = session.dispatch(call_id, world, {"increment"})
        resumed = Session.restore(session.snapshot())
        self.assertEqual(resumed.dispatch(call_id, world, {"increment"}), receipt)
        self.assertEqual(world.counter, 4)
        self.assertEqual(resumed.events, session.events)
        self.assertEqual(resumed.request("double", 5), "demo:2")

    def test_crash_window_is_unknown_and_does_not_rerun(self):
        session, world = Session(), MockWorld()
        call_id = session.request("increment", 7)
        with self.assertRaises(SimulatedCrash):
            session.dispatch(call_id, world, {"increment"}, crash_after_effect=True)
        resumed = Session.restore(session.snapshot())
        self.assertEqual(resumed.events[-1].kind, "started")
        for _ in range(2):
            self.assertEqual(resumed.dispatch(call_id, world, {"increment"}).status, "unknown")
        status, receipts = run_loop(resumed, MockProvider([("increment", 7)]),
                                    world, {"increment"}, 3)
        self.assertEqual((status, len(receipts)), ("unknown", 1))
        self.assertEqual(world.counter, 7)
        self.assertEqual(resumed.events, session.events)

    def test_repeating_provider_stops_at_tool_budget(self):
        session, world = Session(), MockWorld()
        status, receipts = run_loop(session, MockProvider([("increment", 1)], repeat=True),
                                    world, {"increment"}, 3)
        self.assertEqual((status, len(receipts), world.counter), ("budget_exhausted", 3, 3))
        self.assertEqual([e.call_id for e in session.events if e.kind == "requested"],
                         ["demo:1", "demo:2", "demo:3"])
        with self.assertRaises(ValueError):
            run_loop(session, MockProvider([]), world, set(), -1)

    def test_compaction_diagnoses_dropped_constraint(self):
        rule = "Do not increment the counter."
        session = Session(constraints=(rule,))
        history = [rule, "User asked to double 3.", "Result was 6."]
        summary, missing = compact_context(history, session.constraints, lambda _: "Result was 6.")
        self.assertEqual((summary, missing), ("Result was 6.", (rule,)))
        self.assertEqual(Session.restore(session.snapshot()).constraints, (rule,))
        _, missing = compact_context(history, session.constraints, lambda lines: " ".join(lines))
        self.assertEqual(missing, ())

    def test_snapshot_is_stable_and_rejects_unknown_version(self):
        session = Session()
        snapshot = session.snapshot()
        session.request("double", 1)
        self.assertEqual(snapshot.events, ())
        with self.assertRaises(ValueError):
            Session.restore(replace(snapshot, version=2))

    def test_unknown_tools_and_call_ids_have_no_effect(self):
        session, world = Session(), MockWorld()
        call_id = session.request("shell", 1)
        self.assertEqual(session.dispatch(call_id, world, {"shell"}).status, "denied")
        saved_events = session.events
        with self.assertRaises(ValueError):
            session.dispatch("missing:1", world, {"increment"})
        self.assertEqual((world.counter, session.events), (0, saved_events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
