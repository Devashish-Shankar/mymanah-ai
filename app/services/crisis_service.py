from email.mime import text
import re

from app.models.crisis import CrisisModel


class CrisisService:

    def __init__(self):
        self.model = CrisisModel()

    def _has_explicit_denial(self, text: str) -> bool:
        patterns = [
            r"\bnot suicidal\b",
            r"\bnot thinking about suicide\b",
            r"\bno intention of hurting myself\b",
            r"\bno intention to hurt myself\b",
            r"\bdo not want to hurt myself\b",
            r"\bdon't want to hurt myself\b",
            r"\bdo not want to die\b",
            r"\bdon't want to die\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def _is_third_person(self, text: str) -> bool:
        patterns = [
            r"\bmy friend\b",
            r"\bmy brother\b",
            r"\bmy sister\b",
            r"\bmy mother\b",
            r"\bmy father\b",
            r"\bmy colleague\b",
            r"\bsomeone I know\b",
            r"\bthey are suicidal\b",
            r"\bthey wanted to kill themselves\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def _is_historical(self, text: str) -> bool:
        patterns = [
            r"\blast year\b",
            r"\byears ago\b",
            r"\bin the past\b",
            r"\bused to\b",
            r"\bpreviously\b",
            r"\bhistorically\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def _has_high_risk_intent(self, text: str) -> bool:
        patterns = [
            r"\bhave a plan\b",
            r"\bhas a plan\b",
            r"\bplan to kill myself\b",
            r"\bplan to hurt myself\b",
            r"\bplanned to kill myself\b",
            r"\bgoing to kill myself\b",
            r"\bgoing to hurt myself\b",
            r"\bkill myself tonight\b",
            r"\bhurt myself tonight\b",
            r"\battempt(ed)? suicide\b",
            r"\battempt(ed)? to kill myself\b",
            r"\battempt(ed)? to hurt myself\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def _has_ideation(self, text: str) -> bool:
        patterns = [
            r"\bthinking about suicide\b",
            r"\bthinking about killing myself\b",
            r"\bthink about killing myself\b",
            r"\bthinking about ending my life\b",
            r"\bthink about ending my life\b",
            r"\bwant to die\b",
            r"\bdon't want to be alive\b",
            r"\bdo not want to be alive\b",
            r"\bdon't want to wake up\b",
            r"\bdo not want to wake up\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def predict(self, text: str) -> dict:

        model_result = self.model.predict(text)

        crisis_probability = model_result["crisis_probability"]

        # ---------------------------------------------------------
        # Context / safety checks
        # ---------------------------------------------------------

        explicit_denial = self._has_explicit_denial(text)
        third_person = self._is_third_person(text)
        historical = self._is_historical(text)
        high_risk_intent = self._has_high_risk_intent(text)
        ideation = self._has_ideation(text)

        # ---------------------------------------------------------
        # Risk decision
        # ---------------------------------------------------------

        # Explicit denial should override a vocabulary-driven
        # crisis prediction from the classifier.
        if explicit_denial:
            risk = "LOW"

        # Current explicit plan / attempt / imminent self-harm.
        elif high_risk_intent:
            risk = "HIGH"

        # Third-person or historical references should not
        # automatically become HIGH risk for the journal author.
        elif third_person or historical:
            if crisis_probability >= 0.75:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        # Current suicidal/self-harm ideation without explicit
        # plan/attempt.
        elif ideation:
            risk = "MEDIUM"

        elif crisis_probability >= 0.80:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        return {
            "crisisRisk": risk,
            "confidence": round(crisis_probability, 4),
            "crisisProbability": round(
                crisis_probability,
                4
            ),
            "context": {
                "explicitDenial": explicit_denial,
                "thirdPerson": third_person,
                "historical": historical,
                "highRiskIntent": high_risk_intent,
                "ideation": ideation,
            },
        }