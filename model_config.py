__VERBOSE__ = True
__DEBUGGING__ = True

__MODEL__ = "gpt-4"  # stronger when facing more complex function-call-requiring prompts
__MAX_TOKENS__ = 1024
__TEMPERATURE__ = 0

__USER_DATA_PATH__ = "user_data/cleaned_emily.csv"   # test

__DATA_COLUMN__ =  ["TRANSACTION ID","DATE","TRANSACTION DETAILS","DESCRIPTION","CATEGORY","PAYMENT METHOD","WITHDRAWAL AMT","DEPOSIT AMT"]
__COLS_TO_KEEP__ = [                 "DATE",                      "DESCRIPTION",                            "WITHDRAWAL AMT","DEPOSIT AMT"]

__MAX_DATE_RANGE__ = 15
__MAX_ERROR_TRIAL__ = 2