def add(x, y):
     return x + y
def subtract(x, y):
     return x - y
def  multiply(x, y):
     return x * y
def divide(x, y):
     if y == 0:
         return "Error: Division by zero."
     return x / y
def show_menu():
     print("\nSimple Calculator")
     print("1. Add")
     print("2. Subtract")
     print("3. Multiply")
     print("4. Divide")
     print("5. Exit")
def main():
     while True:
         show_menu()
         choice = input("choose an operation (1-5): ")
         if choice == "5":
             print("Goodbye!")
             break
         if choice in ["1", "2", "3", "4"]:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
            except ValueError:

                     continue
         if choice == "1":
             print(f"Result: {add(num1, num2)}")
         elif choice == "2":
             print(f"Result: {subtract(num1, num2)}")
         elif choice == "3":
             print(f"Result: {multiply(num1, num2)}")
         elif choice == "4":
             print(f"Result: {divide(num1, num2)}")
         else:
             print("invalid choice. Please select 1-5.")
if __name__ == "__main__": main()
     #DEZ, RUN THIS!!pythio