class task:
    def __init__(self, cycle=1, rule_num=2):
        self.cycle = cycle
        self.rule_num = rule_num
        self.step=0

    @property
    def rule(self):
        cycle_no = self.step // self.cycle
        return (cycle_no) % self.rule_num

    @property
    def is_bonus(self):
        if (self.step % self.cycle == 0):
            return True
        else:
            return False

def main():
    t = task()
    for i in range(10):
        print(t.rule)
        t.step += 1

if __name__=='__main__':
    main()
