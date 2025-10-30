# simple skeleton script for the automation
import time
from src.attack.run_attack import run_attack # or what ever the name of the attack script is
    

# The orchestrator for the attacks
def main():
    
    experiments = [
        {"name":"test1","strength":0.1},
        {"name":"test2","strength":0.2},
        # ...
    ]
  

    for exp in experiments:
        print("starting:", exp)
        try:
            result = run_attack(exp)
        except Exception as e:
             # minimal error handling
            result = {"name":exp.get("name"),"status": "error","error": str(e)}
        print("Result:", result)
        


if __name__== "__main__":
    main()


