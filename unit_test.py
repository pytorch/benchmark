def test_get_operators():
    from torchbenchmark.operators_collection import list_operators_by_collection

    print(list_operators_by_collection("default"))
    print(list_operators_by_collection("all"))
    print(list_operators_by_collection("liger"))
