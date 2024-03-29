    # {
    #     "type": "function",
    #     "function": {
    #         'name': 'get_current_temperature',
    #         'description': 'gets the current temperature',
    #         'parameters': {}
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         'name': 'get_current_traffic',
    #         'description': 'gets the current traffic',
    #         'parameters': {}
    #     }
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         'name': 'acquire_history_data',
    #         'description': 'obtains user past and current expenses data from start_date to end_date.',
    #         'parameters': {
    #             'type': 'object',
    #             'properties': {
    #                 'start_date': {
    #                     'type': 'string',
    #                     'description': 'start date in the format of DD-MM-YYYY, for user expense data retrieval'
    #                 },
    #                 'end_date': {
    #                     'type': 'string',
    #                     'description': 'end date in the format of DD-MM-YYYY, for user expense data retrieval'
    #                 }
    #             }
    #         }
    #     }
    # },

MANDATORY_FUNCTION_CALL_TEMPLATE = [
    {
        "type": "function",
        "function": {
            'name': 'acquire_history_data',
            'description': 'obtains user past and current expenses data from start_date to end_date.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'start_date': {
                        'type': 'string',
                        'description': 'start date in the format of DD-MM-YYYY, for user expense data retrieval. If user did not specify any starting date, please provide the date exactly a month ago'
                    },
                    'end_date': {
                        'type': 'string',
                        'description': 'end date in the format of DD-MM-YYYY, for user expense data retrieval. If user did not specify any starting date, please provide the date of today.'
                    }
                }
            }
        }
    }
]


FUNCTION_CALL_TEMPLATE = [
    {
        "type": "function",
        "function": {
            'name': 'record_user_expenses',
            'description': 'if user specified that he/she bought some items, call this function extract expenses information and categorise them, along with their values',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'expense_prediction',
            'description': 'predicts user future expenses based on past expenses data.',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'data_analysis',
            'description': 'call this function when user asks about anything related to the users past expenses data.',
            'parameters': {}
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'regular_chat',
            'description': 'must revert to calling this function when user input is not related to any other given functions',
            'parameters': {}
        }
    }
]