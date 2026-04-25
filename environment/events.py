from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Event:
    id: str
    title: str
    source: str
    sender: str
    body: str
    urgency: str
    relationship_stake: str
    actions: List[Dict]


CANONICAL_EVENTS: List[Event] = [
    Event(
        id="E1_staging",
        title="Staging server down - deploy blocked",
        source="pagerduty",
        sender="PagerDuty",
        body="ALERT: staging-prod-1 disk 98%. Four engineers blocked.",
        urgency="critical",
        relationship_stake="team",
        actions=[
            {"id": "fix_directly", "energy_delta": -0.08, "sprint_delta": 0.20, "relationship_delta": {}},
            {"id": "delegate_oncall", "energy_delta": -0.02, "sprint_delta": 0.10, "relationship_delta": {"rahul": -0.01}},
            {"id": "escalate_infra", "energy_delta": -0.06, "sprint_delta": 0.05, "relationship_delta": {"rahul": -0.02}},
        ],
    ),
    Event(
        id="E2_slack",
        title="Peer pings from Priya and Rahul",
        source="slack",
        sender="Multiple",
        body="Need quick help calls for auth module and API spec.",
        urgency="medium",
        relationship_stake="peer",
        actions=[
            {"id": "async_boundary", "energy_delta": 0.04, "sprint_delta": 0.0, "relationship_delta": {"priya": 0.01, "rahul": 0.01}},
            {"id": "both_calls_now", "energy_delta": -0.20, "sprint_delta": -0.03, "relationship_delta": {"priya": 0.03, "rahul": 0.03}},
            {"id": "ignore_slack", "energy_delta": -0.03, "sprint_delta": -0.02, "relationship_delta": {"priya": -0.04, "rahul": -0.04}},
        ],
    ),
    Event(
        id="E3_client_email",
        title="Client escalation from David Chen",
        source="email",
        sender="david.chen@meridian.com",
        body="Following up again on dashboard latency issue.",
        urgency="high",
        relationship_stake="client",
        actions=[
            {"id": "acknowledge_timeline", "energy_delta": 0.03, "sprint_delta": 0.0, "relationship_delta": {"david_chen": 0.08}},
            {"id": "full_apology", "energy_delta": -0.09, "sprint_delta": -0.05, "relationship_delta": {"david_chen": -0.04}},
            {"id": "cc_manager", "energy_delta": -0.02, "sprint_delta": 0.0, "relationship_delta": {"david_chen": 0.02, "sundar": -0.02}},
        ],
    ),
    Event(
        id="E4_leave",
        title="Thursday leave still pending",
        source="calendar",
        sender="HR System",
        body="Leave requested 3 months ago remains unresolved.",
        urgency="high",
        relationship_stake="manager",
        actions=[
            {"id": "ping_sundar", "energy_delta": 0.10, "sprint_delta": 0.0, "relationship_delta": {"sundar": 0.00}},
            {"id": "wait_hope", "energy_delta": -0.12, "sprint_delta": 0.0, "relationship_delta": {"sundar": 0.0}},
            {"id": "cancel_leave", "energy_delta": -0.25, "sprint_delta": 0.05, "relationship_delta": {"sundar": 0.01}},
        ],
    ),
    Event(
        id="E5_appraisal",
        title="Annual appraisal due Friday",
        source="hr",
        sender="HR Portal",
        body="12 sections, around 90 minutes, impacts promo cycle.",
        urgency="high",
        relationship_stake="hr",
        actions=[
            {"id": "block_wednesday", "energy_delta": 0.03, "sprint_delta": 0.0, "relationship_delta": {}},
            {"id": "rush_tonight", "energy_delta": -0.14, "sprint_delta": 0.0, "relationship_delta": {}},
            {"id": "ask_extension", "energy_delta": -0.03, "sprint_delta": 0.0, "relationship_delta": {}},
        ],
    ),
    Event(
        id="E6_oncall",
        title="Arjun asks on-call coverage again",
        source="slack",
        sender="arjun",
        body="Third request this quarter, asks Wednesday night coverage.",
        urgency="low",
        relationship_stake="peer",
        actions=[
            {"id": "no_clearly_kindly", "energy_delta": 0.12, "sprint_delta": 0.0, "relationship_delta": {"arjun": -0.02}},
            {"id": "yes_people_pleaser", "energy_delta": -0.28, "sprint_delta": -0.15, "relationship_delta": {"arjun": 0.05}},
            {"id": "counter_offer", "energy_delta": 0.02, "sprint_delta": 0.0, "relationship_delta": {"arjun": 0.00}},
        ],
    ),
    Event(
        id="E7_standup",
        title="Optional 10:30 PM standup",
        source="calendar",
        sender="Meridian Zoom",
        body="No direct action items, but visibility pressure.",
        urgency="medium",
        relationship_stake="client",
        actions=[
            {"id": "decline_async", "energy_delta": 0.14, "sprint_delta": 0.0, "relationship_delta": {"david_chen": 0.02}},
            {"id": "attend_sacrifice", "energy_delta": -0.30, "sprint_delta": -0.10, "relationship_delta": {"david_chen": 0.03}},
            {"id": "delegate_teammate", "energy_delta": 0.05, "sprint_delta": 0.0, "relationship_delta": {"priya": -0.02}},
        ],
    ),
]
