#concession stand program

import json
import datetime

def show_receipt():
    file_path="concession_stand_receipt.json"
    try:
        with open(file_path,"r") as file:
                data = json.load(file)
                print(data)
    except FileNotFoundError:
        print("Receipt file does not exist yet.")



def display_menu(menu):
    for key,value in menu.items():
        print(f"{key:<20}  {value:>2}")


def select_item(menu):
    orders=[]  
    value=[]
    quantities=[]
    amounts=[]

    while True:
        item=input("select an item(q to quit)🍽:")
        item=item.title()

        if item=="Q": 
            break

        elif item not in menu:
            print("item not in menu.PLEASE TRY AGAIN")
            

        else:
            while True:
                try:
                    qty=int(input(f"enter the quantity of {item} you want:"))
                    if qty<=0:
                        print("enter positive value of quantity")
                        continue
                    orders.append(item)
                    prices=menu.get(item)
                    value.append(prices)
                    quantities.append(qty)
                    amount = prices * qty
                    amounts.append(amount)
                    break
                except ValueError:
                    print("quantity can not be string.")
                    
    return orders,quantities,amounts,value

def coupon():
    coupon_ask=input("Do you have coupon?(yes or no)").strip().lower()
    if coupon_ask=="yes":
        coupon_code=input("enter the coupon code:")
        if coupon_code=="SS1967":
            print("valid coupon you can avail discount of 10%😃")
            return True
        else:
            print("invalid coupon code😔")
            return False
    return False
    

def calculate_discount(amounts):
    return sum(amounts) * 0.10


def calculate_total(amounts, has_coupon):
    if has_coupon:
        return sum(amounts) - calculate_discount(amounts)
    return sum(amounts)

def receipt(orders,quantities,amounts,has_coupon,value):
    print(f"{'item':<20} {'quantity':<20} {'unit price':<20} {'price':>10}")
    print('-'*75)

    for ite,quantitie,amt,val in zip(orders,quantities,amounts,value): 
        print(f"{ite:<20} {quantitie:<20} {val:<20} {amt:>10}")
    print('-' * 75)
    if has_coupon:
        discount = calculate_discount(amounts)
        total = calculate_total(amounts, has_coupon)
        print(f"{'discount':<20} {'10%':<20} {discount:>20}")
    else:
        total = calculate_total(amounts, has_coupon)
    print(f"{'TOTAL':<20} {round(total,2):>50}")


def time_date():
     
    date=datetime.date.today()
    print(date)
    now=datetime.datetime.now()
    print(now.time())
    return date,now


def save_receipt( orders, quantities, amounts, has_coupon, value, date, now):
    file_path="concession_stand_receipt.json"
    total = calculate_total(amounts, has_coupon)
    receipt_data={"items":orders,"quantity":quantities,"unit price":value,"price":amounts,"coupon":has_coupon,"date":str(date),"time":now.strftime("%H:%M:%S"),"total":total}

    with open(file_path,"w") as file:
        json.dump(receipt_data, file, indent=2)

    print("✅ Receipt saved successfully.")


    
def main(): 
    menu={ "Popcorn":300 ,
            "Pizza":500 ,
            "Sandwich" :150 , 
            "Nachos And Cheese":350,
            "Hot Dog":150,
            "Candy":60,
            "Mineral Water":50,
            "Coffee":100,
            "Hot Chocolate":110,
            "Soda":70,
            "Smoothie":170}
    show_receipt()
    print("WELCOME TO THE CONCESSION STAND!!")
    print()
    print("-----------MENU📜-------------")
    display_menu(menu)
    print("----------------------------")  
    orders, quantities, amounts,value = select_item(menu)
    has_coupon = coupon()
    print()
    print("------------------------------YOUR RECEIPT---------------------------------------")
    receipt(orders,quantities,amounts,has_coupon,value) 
    print("---------------------------------------------------------------------------------")
    date,now=time_date()
    save_receipt(orders,quantities,amounts,has_coupon,value,date,now)

if __name__=="__main__":
    main()