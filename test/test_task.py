import task

def test_task():

    print("")
    result = ""
    t=task.task()
    for i in range(10):
        result+=str(t.rule)
        t.step += 1
    print(result)
    assert result=="0101010101"

    result = ""
    t=task.task(3)
    for i in range(10):
        result+=str(t.rule)
        t.step += 1
    print(result)
    assert result=="0001110001"

    result = ""
    t=task.task(2,3)
    for i in range(10):
        result+=str(t.rule)
        t.step += 1
    print(result)
    assert result=="0011220011"

    result = ""
    t=task.task(3,3)
    for i in range(10):
        result+=str(t.rule)
        t.step += 1
    print(result)
    assert result=="0001112220"

if __name__=='__main__':
    test_task()
