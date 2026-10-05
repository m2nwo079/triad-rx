import json
from pathlib import Path

import pytest

from triadrx.assignees import AssigneeClassifier


@pytest.fixture(scope="module")
def clf():
    cfg = json.loads(Path("config/study.json").read_text())["applied"]["assignee_classes"]
    return AssigneeClassifier(cfg)


@pytest.mark.parametrize("code,name,expected", [
    ("2", "DIVERGENT TECHNOLOGIES, INC.", "company"),
    ("3", "Samsung Electronics Co., Ltd.", "company"),
    ("12", "Google LLC", "company"),
    ("2", "Stanford University", "academic"),
    ("3", "Korea Advanced Institute of Science and Technology", "ambiguous"),
    ("2", "Massachusetts Institute of Technology", "academic"),
    ("3", "Technische Universität München", "academic"),
    ("2", "University Research Corporation", "ambiguous"),
    ("2", "Allen Institute", "ambiguous"),
    ("2", "XYZ Research Corporation", "company"),
    ("4", "", "individual"),
    ("5", None, "individual"),
    ("6", "The United States of America as represented by the Secretary of the Navy", "government"),
    ("7", "Fraunhofer", "government"),
    ("7", "NANKAI UNIVERSITY", "academic"),
    ("6", "University of Cincinnati", "academic"),
    ("7", "SHANGHAI INSTITUTE OF TECHNOLOGY", "academic"),
    ("7", "CHINESE PLA GENERAL HOSPITAL", "ambiguous"),
    ("7", "STATE GRID ZHEJIANG ELECTRIC POWER COMPANY LIMITED", "company"),
    ("7", "CENTRE NATIONAL DE LA RECHERCHE SCIENTIFIQUE", "government"),
    ("9", "University of South Florida", "academic"),
    ("1", "Someone", "unknown"),
    ("", "Someone", "unknown"),
    (None, "Someone", "unknown"),
    ("2.0", "Acme Corp", "company"),
    ("2", "Agri Agency", "company"),
])
def test_classify(clf, code, name, expected):
    assert clf.classify(code, name) == expected


def test_suffix_needs_whole_word(clf):
    # "AG" must not match inside "Agency"
    assert not clf.suffix.search("Agency")
    assert clf.suffix.search("Siemens AG")
