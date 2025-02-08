# Diet plans covering all goals and diet types
diet_plans = {
    "weight_loss": {
        "vegetarian": {
            "Breakfast": ("Oats with almond milk, chia seeds, banana", "300 kcal, 10g protein, 45g carbs, 5g fats"),
            "Lunch": ("Grilled tofu salad with quinoa", "450 kcal, 25g protein, 55g carbs, 12g fats"),
            "Dinner": ("Lentil soup with whole wheat toast", "400 kcal, 20g protein, 50g carbs, 8g fats"),
            "Cheat Meal": ("Dark chocolate and mixed nuts", "250 kcal, 5g protein, 20g carbs, 15g fats")
        },
        "non_vegetarian": {
            "Breakfast": ("Scrambled eggs with whole wheat toast", "350 kcal, 22g protein, 40g carbs, 10g fats"),
            "Lunch": ("Grilled chicken with steamed veggies", "500 kcal, 40g protein, 50g carbs, 8g fats"),
            "Dinner": ("Salmon with quinoa and greens", "550 kcal, 42g protein, 55g carbs, 10g fats"),
            "Cheat Meal": ("Greek yogurt with honey and nuts", "280 kcal, 15g protein, 35g carbs, 8g fats")
        },
        "vegan": {
            "Breakfast": ("Smoothie with spinach, banana, almond milk, and flaxseeds", "320 kcal, 12g protein, 50g carbs, 5g fats"),
            "Lunch": ("Chickpea salad with avocado", "480 kcal, 22g protein, 55g carbs, 12g fats"),
            "Dinner": ("Stir-fried tofu with brown rice", "520 kcal, 25g protein, 60g carbs, 10g fats"),
            "Cheat Meal": ("Homemade energy balls with dates and nuts", "300 kcal, 8g protein, 40g carbs, 12g fats")
        }
    },
    "muscle_gain": {
        "vegetarian": {
            "Breakfast": ("Paneer-stuffed paratha with yogurt", "550 kcal, 30g protein, 60g carbs, 15g fats"),
            "Lunch": ("Rajma with brown rice", "600 kcal, 35g protein, 65g carbs, 12g fats"),
            "Dinner": ("Stir-fried tofu with quinoa", "580 kcal, 30g protein, 60g carbs, 10g fats"),
            "Cheat Meal": ("Coconut energy bars", "350 kcal, 12g protein, 40g carbs, 15g fats")
        },
        "non_vegetarian": {
            "Breakfast": ("Chicken omelette with whole wheat toast", "600 kcal, 40g protein, 50g carbs, 12g fats"),
            "Lunch": ("Grilled chicken with sweet potato", "700 kcal, 55g protein, 70g carbs, 10g fats"),
            "Dinner": ("Steak with roasted vegetables", "750 kcal, 60g protein, 50g carbs, 15g fats"),
            "Cheat Meal": ("Protein shake with peanut butter", "400 kcal, 30g protein, 35g carbs, 12g fats")
        },
        "vegan": {
            "Breakfast": ("Chia pudding with soy milk and berries", "500 kcal, 20g protein, 55g carbs, 10g fats"),
            "Lunch": ("Lentil and quinoa bowl", "650 kcal, 35g protein, 70g carbs, 12g fats"),
            "Dinner": ("Tofu stir-fry with wild rice", "700 kcal, 38g protein, 65g carbs, 14g fats"),
            "Cheat Meal": ("Almond butter on whole wheat toast", "350 kcal, 12g protein, 40g carbs, 10g fats")
        }
    },
    "balanced_diet": {
        "vegetarian": {
            "Breakfast": ("Whole grain toast with peanut butter", "350 kcal, 12g protein, 40g carbs, 10g fats"),
            "Lunch": ("Mixed vegetable curry with brown rice", "550 kcal, 25g protein, 60g carbs, 15g fats"),
            "Dinner": ("Dal with chapati and salad", "500 kcal, 20g protein, 55g carbs, 10g fats"),
            "Cheat Meal": ("Homemade fruit smoothie", "300 kcal, 8g protein, 45g carbs, 5g fats")
        },
        "non_vegetarian": {
            "Breakfast": ("Boiled eggs with whole grain toast", "400 kcal, 25g protein, 40g carbs, 8g fats"),
            "Lunch": ("Grilled fish with quinoa and salad", "600 kcal, 45g protein, 60g carbs, 10g fats"),
            "Dinner": ("Chicken curry with brown rice", "650 kcal, 50g protein, 70g carbs, 12g fats"),
            "Cheat Meal": ("Yogurt parfait with granola", "320 kcal, 15g protein, 50g carbs, 6g fats")
        },
        "vegan": {
            "Breakfast": ("Avocado toast on whole grain bread", "380 kcal, 10g protein, 50g carbs, 12g fats"),
            "Lunch": ("Chickpea and spinach stew with brown rice", "570 kcal, 28g protein, 65g carbs, 14g fats"),
            "Dinner": ("Mushroom stir-fry with quinoa", "580 kcal, 30g protein, 60g carbs, 12g fats"),
            "Cheat Meal": ("Dark chocolate with almonds", "250 kcal, 5g protein, 20g carbs, 15g fats")
        }
    }
}

# Numeric menu options
goal_options = {
    "1": "weight_loss",
    "2": "muscle_gain",
    "3": "balanced_diet"
}

diet_options = {
    "1": "vegetarian",
    "2": "non_vegetarian",
    "3": "vegan"
}

def diet_chart_generator():
    print("\nChoose your goal:")
    for key, value in goal_options.items():
        print(f"{key}. {value.replace('_', ' ').title()}")
    goal = input("Enter choice (1/2/3): ")
    
    if goal not in goal_options:
        print("❌ Invalid choice! Please try again.")
        return
    
    print("\nChoose your diet type:")
    for key, value in diet_options.items():
        print(f"{key}. {value.replace('_', ' ').title()}")
    diet_type = input("Enter choice (1/2/3): ")
    
    if diet_type not in diet_options:
        print("❌ Invalid choice! Please try again.")
        return

    goal = goal_options[goal]
    diet_type = diet_options[diet_type]

    try:
        plan = diet_plans[goal][diet_type]
        print("\n--- Your Diet Plan ---")
        for meal, (food, nutrition) in plan.items():
            print(f"{meal}: {food} | {nutrition}")
        print("----------------------\n")
    except KeyError:
        print("\n❌ Sorry, this combination is not available. Please try a different option.")

diet_chart_generator()
