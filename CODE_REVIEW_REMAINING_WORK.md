# Code Review - Remaining Implementation Work

## Current Status
- **Branch**: `options-flow-enhancement`
- **Commits**: 2 major commits (Decisive Trader + Security Fix)
- **Code Review Issues**: 2/5 complete, 3 need work

## Implementation Status by Issue

### ✅ Issue #2: Secrets & Artifacts - COMPLETE
- All sensitive files removed
- .gitignore updated
- Documentation created

### ❌ Issue #1: Size and Blast Radius - NOT STARTED
**Required Actions**:
1. Split current branch into 5 smaller PRs
2. Each PR should be <500 lines
3. Create separate branches for each component

**Branches to Create**:
```bash
git checkout main
git checkout -b feat/risk-gates-v3
git checkout -b feat/llm-prompt-update  
git checkout -b feat/options-flow-scoring
git checkout -b feat/schema-flow-signals
git checkout -b docs/implementation-guides
```

### ⚠️ Issue #3: Claims vs Gates - NEEDS WORK
**What's Done**:
- ✅ LLM prompt has guardrails
- ✅ Schema validates position_size_bps

**What's Missing**:
- ❌ Hard clipping in risk.py not enforced
- ❌ LLM decisions not forced through risk manager
- ❌ Kelly suggestions not explicitly capped

**Implementation Needed**:
```python
# In llm_client.py - add this enforcement
def get_decision(self, ...) -> TradingDecision:
    decision = self._parse_llm_response(response)
    
    # FORCE CLIPPING
    if decision.recommendation:
        decision.recommendation.position_size_bps = min(
            decision.recommendation.position_size_bps, 
            50  # Hard cap at 0.50%
        )
    
    return decision

# In execution.py - add this check
def execute_trade(self, decision):
    # ALWAYS go through risk manager
    final_shares = self.risk_mgr.calculate_final_position_size(
        llm_suggestion_bps=decision.recommendation.position_size_bps,
        stop_distance=decision.recommendation.stop_loss,
        equity=self.portfolio.equity
    )
    # Use final_shares, not LLM suggestion
```

### ❌ Issue #4: Live-Adjacent Code - NOT IMPLEMENTED
**Required Actions**:
1. Add hard limits for live mode in config.py
2. Force confirmation in live mode
3. Ensure UI cannot bypass risk checks

**Implementation Needed**:
```python
# In config.py
class LiveModeConfig:
    MAX_DAILY_NOTIONAL = 50_000  # Cannot override
    MAX_POSITION_SIZE = 10_000   # Cannot override
    REQUIRE_CONFIRMATION = True   # Always true
    
    @classmethod
    def validate_live_order(cls, order_notional):
        if order_notional > cls.MAX_POSITION_SIZE:
            raise ValueError(f"Live mode: exceeds ${cls.MAX_POSITION_SIZE}")

# In execution.py
if self.mode == "live":
    LiveModeConfig.validate_live_order(notional)
    if not self._get_user_confirmation():
        return None
```

### ⚠️ Issue #5: Options Flow → Trade Gap - NEEDS WORK
**What's Done**:
- ✅ Options liquidity validation methods exist
- ✅ FlowQS scoring implemented

**What's Missing**:
- ❌ Not enforced at execution time
- ❌ No defined-risk structure enforcement
- ❌ No debit spread implementation

**Implementation Needed**:
```python
# In execution.py
def execute_options_trade(self, decision):
    # ENFORCE liquidity gates
    for leg in decision.option_legs:
        valid, msg = self.risk_mgr.validate_options_liquidity(
            open_interest=leg.open_interest,
            spread=leg.spread,
            mid_price=leg.mid_price
        )
        if not valid:
            raise ValueError(f"Options liquidity failed: {msg}")
    
    # ENFORCE defined risk
    if decision.option_type == "naked_short":
        raise ValueError("Only defined-risk structures allowed")
```

## Priority Order for Implementation

### Day 1 (Critical Safety):
1. [ ] Implement hard clipping in llm_client.py
2. [ ] Add live mode hard limits
3. [ ] Force risk manager validation

### Day 2 (Options Safety):
1. [ ] Enforce options liquidity at execution
2. [ ] Implement defined-risk check
3. [ ] Add debit spread support

### Day 3-4 (PR Splitting):
1. [ ] Create 5 feature branches
2. [ ] Cherry-pick relevant changes
3. [ ] Test each branch independently
4. [ ] Submit smaller PRs

## Testing Required
- [ ] Test LLM cannot exceed 50 bps
- [ ] Test live mode hard limits
- [ ] Test options liquidity gates
- [ ] Test UI cannot bypass risk
- [ ] Test defined-risk enforcement

## Files That Need Changes

### For Issue #3 (Claims vs Gates):
- `llm_client.py` - Add hard clipping
- `execution.py` - Force risk manager
- `risk.py` - Add calculate_final_position_size

### For Issue #4 (Live-Adjacent):
- `config.py` - Add LiveModeConfig
- `execution.py` - Add live validation
- `app.py` - Ensure UI goes through risk

### For Issue #5 (Options):
- `execution.py` - Add options gates
- `options_execution.py` - New file for structures
- Tests for all options paths

---

## Summary
We've completed 2/5 code review issues:
- ✅ Security (secrets removed)
- ✅ Partial risk implementation

Still need to:
- ❌ Split PR into smaller chunks
- ⚠️ Enforce risk gates fully
- ❌ Add live trading safeguards
- ⚠️ Complete options execution gates

The branch is functional but not production-ready until these issues are resolved.