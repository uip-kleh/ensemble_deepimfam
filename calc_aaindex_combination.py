from new_deepimfam.component import AAindex1

if __name__ == "__main__":
    aaindex1 = AAindex1(config_path="new_deepimfam/config.yaml")
    aaindex1.calc()
    aaindex1.disp("NAKH900107", "PALJ810108")