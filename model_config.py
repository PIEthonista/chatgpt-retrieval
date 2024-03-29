__VERBOSE__ = False
__MODEL__ = "gpt-4"
__MAX_TOKENS__ = 1024
__TEMPERATURE__ = 0

__USER_DATA_PATH__ = "user_data/cleaned_emily.csv"   # test

__DATA_COLUMN__ =  ["TRANSACTION ID","DATE","TRANSACTION DETAILS","DESCRIPTION","CATEGORY","PAYMENT METHOD","WITHDRAWAL AMT","DEPOSIT AMT"]
__COLS_TO_KEEP__ = [                 "DATE",                      "DESCRIPTION",                            "WITHDRAWAL AMT","DEPOSIT AMT"]