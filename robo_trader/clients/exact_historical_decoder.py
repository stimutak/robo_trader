"""Preserve historical volume at the pinned ib_async wire boundary."""

from decimal import Decimal, InvalidOperation

from ib_async import IB
from ib_async.decoder import Decoder
from ib_async.objects import BarData
from ib_async.util import parseIBDatetime

from robo_trader.utils import ibkr_safe  # noqa: F401 - explicit disconnect guard


class ExactHistoricalDecoder(Decoder):
    """Decode complete historical responses atomically with decimal volumes.

    This adapts message 17 in ib_async 2.0.1. It does not infer volume units or
    change the streaming historical-update decoder, which the worker never requests.
    """

    def historicalData(self, fields):
        # Establish request ownership before touching a pending request.
        if len(fields) < 2 or type(fields[1]) is not str or not fields[1].isdigit():
            raise ValueError("Historical response request identity is malformed")
        req_id = int(fields[1])
        try:
            if len(fields) < 5 or fields[0] != "17" or not fields[4].isdigit():
                raise ValueError("Historical response header is malformed")
            count = int(fields[4])
            if len(fields) != 5 + count * 8:
                raise ValueError("Historical response bar count does not match payload")
            rows = []
            for offset in range(5, len(fields), 8):
                date, opened, high, low, close, raw_volume, average, bar_count = fields[
                    offset : offset + 8
                ]
                parseIBDatetime(date)  # Validate before any wrapper callback publishes a row.
                volume = Decimal(raw_volume)
                if not volume.is_finite() or volume < 0:
                    raise ValueError("Historical source volume must be finite and nonnegative")
                rows.append(
                    BarData(
                        date=date,
                        open=float(opened),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=volume,
                        average=float(average),
                        barCount=int(bar_count),
                    )
                )
        except (ValueError, TypeError, InvalidOperation, OverflowError, OSError, KeyError) as exc:
            # ib_async otherwise swallows decoder errors, then returns an empty
            # list after its request timeout. Complete this request as a failure.
            self.wrapper._results.pop(req_id, None)
            self.wrapper._endReq(
                req_id, ValueError(f"Invalid historical response: {exc}"), success=False
            )
            return
        for bar in rows:
            self.wrapper.historicalData(req_id, bar)
        self.wrapper.historicalDataEnd(req_id, fields[2], fields[3])


class ExactHistoricalIB(IB):
    """Install the exact historical decoder before any connection is possible."""

    def __init__(self):
        super().__init__()
        self.client.decoder = ExactHistoricalDecoder(self.wrapper, 0)
