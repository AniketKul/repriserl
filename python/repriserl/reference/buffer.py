"""Bounded single-process reference store. Put only after the first update commits."""
from collections import Counter
from dataclasses import dataclass
from random import Random
from threading import RLock

from .types import PutResult, ReserveRequest, Reservation, RolloutGroup, _count


@dataclass(slots=True)
class _BatchRecord:
    request: ReserveRequest
    reservation: Reservation
    state: str = "pending"
    applied: tuple[str, ...] | None = None


class ReferenceReplayStore:
    """One learner and epoch. Capacity exhaustion is explicit backpressure.

    Age is measured at admission, 1 <= current step - generating step <= max_age.
    Groups start with one committed fresh use. A reservation holds one extra use
    per candidate until applied or released. IDs/receipts live for this store's
    lifetime so an old retry cannot double-apply. No on-disk recovery is implied.
    """

    def __init__(self, capacity: int = 32_768, *, epoch: int = 0, group_size: int = 4,
                 max_age: int = 8, max_uses: int = 3, max_batches: int = 65_536):
        for name, val, minimum in (("capacity", capacity, 1), ("epoch", epoch, 0),
                                   ("group_size", group_size, 2), ("max_age", max_age, 1),
                                   ("max_uses", max_uses, 1), ("max_batches", max_batches, 1)):
            _count(val, name, minimum)
        self._capacity, self._epoch, self._group_size = capacity, epoch, group_size
        self._max_age, self._max_uses, self._max_batches = max_age, max_uses, max_batches
        self._groups: dict[str, RolloutGroup] = {}
        self._uses: Counter[str] = Counter()
        self._pending: Counter[str] = Counter()
        self._records: dict[str, _BatchRecord] = {}
        self._lock = RLock()

    def put_group(self, group: RolloutGroup) -> PutResult:
        if not isinstance(group, RolloutGroup) or len(group.rollouts) != self._group_size:
            raise ValueError("expected a complete rollout group")
        with self._lock:
            if group.group_id in self._groups:
                if group != self._groups[group.group_id]:
                    raise ValueError("conflicting payload for group ID")
                return PutResult(False)
            if len(self._groups) >= self._capacity:
                raise BufferError("group capacity exhausted")
            self._groups[group.group_id] = group
            self._uses[group.group_id] = 1
            return PutResult(True)

    def get_group(self, group_id: str) -> RolloutGroup:
        with self._lock:
            return self._groups[group_id]

    def use_count(self, group_id: str) -> int:
        with self._lock:
            if group_id not in self._groups:
                raise KeyError(group_id)
            return self._uses[group_id]

    def _check_epoch(self, epoch):
        if type(epoch) is not int or epoch != self._epoch:
            raise ValueError("stale or invalid epoch")

    def reserve(self, request: ReserveRequest) -> Reservation:
        with self._lock:
            self._check_epoch(request.epoch)
            if request.batch_id in self._records:
                record = self._records[request.batch_id]
                if record.request != request or record.state == "released":
                    raise ValueError("conflicting or released batch identity")
                return record.reservation
            if len(self._records) >= self._max_batches:
                raise BufferError("batch receipt capacity exhausted")
            policy = request.current_policy
            pools = {name: [] for name in request.quotas}
            for group_id, group in self._groups.items():
                if (group.stratum in pools and group.policy.lineage == policy.lineage
                    and 1 <= policy.step - group.policy.step <= self._max_age
                    and self._uses[group_id] + self._pending[group_id] < self._max_uses):
                    pools[group.stratum].append(group_id)
            rng = Random(request.seed)
            selected = []
            unmet = {}
            for name, quota in sorted(request.quotas.items()):
                pool = sorted(pools[name])
                take = min(len(pool), quota)
                selected.extend(rng.sample(pool, take))
                unmet[name] = quota - take
            reservation = Reservation(request.batch_id, tuple(selected), unmet)
            self._pending.update(selected)
            self._records[request.batch_id] = _BatchRecord(request, reservation)
            return reservation

    def record_applied(self, batch_id: str, epoch: int, accepted_ids: tuple[str, ...] | None = None) -> None:
        """Commit only accepted candidates, release every rejected candidate."""
        with self._lock:
            self._check_epoch(epoch)
            record = self._records[batch_id]
            candidates = record.reservation.group_ids
            accepted = candidates if accepted_ids is None else tuple(accepted_ids)
            if len(accepted) != len(set(accepted)) or not set(accepted).issubset(candidates):
                raise ValueError("accepted IDs must be distinct reserved candidates")
            if record.state == "applied" and record.applied == accepted:
                return
            if record.state != "pending":
                raise ValueError("batch was released or committed with a different selection")
            self._uses.update(accepted)
            self._pending.subtract(candidates)
            record.state, record.applied = "applied", accepted

    def release(self, reservation_id: str, epoch: int) -> None:
        with self._lock:
            self._check_epoch(epoch)
            record = self._records[reservation_id]
            if record.state == "applied":
                raise ValueError("cannot release applied batch")
            if record.state == "pending":
                self._pending.subtract(record.reservation.group_ids)
                record.state = "released"
