#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class GenericQuery(object):
    """
    This class serves as a template for the queries.
    Each query class has to inherit this class.

    """

    def __init__(self):
        print("Generic Query")

    def call_planner(self, domain_file, problem_file, result_file):
        print("Call the planner")
