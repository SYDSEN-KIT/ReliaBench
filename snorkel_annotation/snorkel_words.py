import re
import numpy as np

LABELS = [
    "State Deviation",
    "Process Flow",
    "Maintenance",
    "Operational",
    "Structural",
    "Temporal and Probabilistic"
]

def weak_label(text):

    text = str(text).lower()

    y = np.zeros(len(LABELS), dtype=int)

    # =================================================
    # 0 State Deviation
    # =================================================

    if contains_keyword(text, [
        "failure", "fault", "malfunction", "error",
        "breakdown",  "failed", "defect", "crash",
        "fail",  "stall", "stalled", "stalling", 
        "shutdown", "outage", "crashed", "quit",
        "collapsed", "exploded", "ruptured", "fractured", "cracked", "crack", "cracks", 
        "misfire", "misfired", "fails", "jammed", "jam", "jams", "stuck", "stucks", "faulted", 
        "abort", "termination", "interrupt", "interrupted", "interrupts",
        "broken",  "aborted",  "halted", "rolled over",
    ]):
        y[0] = 1

    # =================================================
    # 1 Process Flow
    # =================================================

    if contains_keyword(text, [
        "before", "after", "during", 
        "then", "prior to", 
        "later", "at the same time", "followed by", "finally", "in parallel",
        "began", "begin", "started", "continued",
        "completed", "preceded", "complete", "start", "end", "ended", "first", "last", "finish", "finished",
        "led to", "resulted in", "triggered", "caused", "leads to", "trigger", "cause", "results in",
        "as a result", "causes", 
        "induced", "induces",  "initiates",
        "provoked", "initiated", "provoke", "provokes", "causing",
        "break", "stopped","stop", "halt",
    ]):
        y[1] = 1

    # =================================================
    # 2 Maintenance
    # =================================================

    if contains_keyword(text, [
        "repair", "replaced", "replace", "repaired", "repairs",
        "overhaul", "serviced", "service", "inspect",
        "fixed", "changed", "calibrated", "calibrate", "change", "fix", "restore", "restored", "recalibrated",
        "adjusted", "cleaned", "lubricated", "adjust", "clean", "lubricate", 
        "restored", "maintain", "maintained", "maintains", "maintaining", 
        "wear", "degraded", "degrade", "adjustment", "aging",
    ]):
        y[2] = 1

    # =================================================
    # 3 Operational
    # =================================================

    if contains_keyword(text, [
        "generate", "generates", "produce", 
        "produced", "generated",
        "load", "capacity", "efficiency", "flow rate", "runtime",
        "throughput", "output", "response time", "latency", "utilization",
        "temperature", "pressure", "voltage", "humidity",
        "speed", "rpm", "torque", "consumption", "drained",
        "environmental", "weather",
        "conditions", "condition",
        "environment", "setting", "settings", "config", "configuration",
        "mode", "state",
        "overheat", "signal", 
        "active", "running", "operating", "waiting", "inactive",
        "standby", "idle", "nominal", "inoperative", "in service",
        "limits", "threshold", "thresholds",
        "parameter",
        "increases", "decreases",
        "affects", "influences", "depends on", "varies with", "changes with",
    ]):
        y[3] = 1

    # =================================================
    # 4 Structural
    # =================================================

    if contains_keyword(text, [
        "group", "grouped", "category", "groups", "joins", "joined", "join",
        "consists of", "composed of", "includes", "include", "next to",
        "part of", "belongs to", "contain", "contains", "contained", "connects",
        "connected to", "connect",
        "linked to", "links",
        "integrated with", "integrates", 
        "member of", "as one", "together", "component of", "module", "subsystem",
    ]):
        y[4] = 1

    # =================================================
    # 5 Temporal and Probabilistic
    # =================================================

    if contains_keyword(text, [
        "minute", "hour", "day", "seconds", "minutes", "hours", "days", "month", "second",
        "m/s", "weeks", "week", "year", "years",
        "frequency", "duration", "time", "useful life", 
        "period", "periodic",
        "probability", "probabilistic", "probabilistically",
        "uncertainty", "likelihood", "confidence", "interval", "distribution",
        "gaussian", "mean", "median", "variance", "standard deviation",
        "expected value", "bayesian", "rate", "function",
        "estimate", "estimated", "estimates", "monte carlo", "prediction",
    ]):
        y[5] = 1

    # =================================================
    # all-zero vector means NONE
    # =================================================

    return y

def contains_keyword(text, keywords):
    text = str(text).lower()

    for kw in keywords:
        kw = kw.lower()

        pattern = r"\b" + re.escape(kw) + r"\b"

        if re.search(pattern, text):
            return True

    return False