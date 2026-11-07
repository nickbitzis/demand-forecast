"""The made-up restaurant group: locations, venues, menu, recipes, ingredients, and suppliers.

Ridgeline Bowl Co. runs three brick and mortar sites in Denver and one food truck. Coordinates are kilometers on a
flat city grid with downtown at (0, 0). Everything here is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 15-minute slots from 10:00 to 22:00. The stores open 10:30 to 21:30, so the first and last two slots are zero.
SLOT_START_HOUR = 10.0
N_SLOTS = 48
OPEN_SLOT, CLOSE_SLOT = 2, 46


def slot_hour(slot: int) -> float:
    return SLOT_START_HOUR + slot * 0.25


def slot_label(slot: int) -> str:
    h = slot_hour(slot)
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    x: float
    y: float
    base_daily: float  # item units on an ordinary Tuesday in the first year
    channels: dict  # base channel shares: dine_in, patio, takeout, delivery
    lunch_weight: float  # how lunch-heavy the intraday profile is (0 to 1)
    mobile: bool = False
    opens_days_before_end: int | None = None  # cold start: opens this many days before the last history day
    analog: str | None = None  # similar location used for cold start
    prior_ratio: float = 1.0  # expected level relative to the analog before any sales are seen


LOCATIONS: list[Location] = [
    Location(
        "lodo",
        "LoDo",
        0.0,
        0.0,
        620,
        {"dine_in": 0.37, "patio": 0.33, "takeout": 0.18, "delivery": 0.12},
        0.68,
    ),
    Location(
        "cap-hill",
        "Capitol Hill",
        2.2,
        -1.5,
        540,
        {"dine_in": 0.40, "patio": 0.06, "takeout": 0.22, "delivery": 0.32},
        0.48,
    ),
    Location(
        "highlands",
        "Highlands",
        -2.4,
        1.8,
        470,
        {"dine_in": 0.42, "patio": 0.18, "takeout": 0.20, "delivery": 0.20},
        0.52,
        opens_days_before_end=68,
        analog="cap-hill",
        prior_ratio=0.85,
    ),
    Location(
        "truck",
        "Food truck",
        0.0,
        0.0,
        110,
        {"dine_in": 0.0, "patio": 0.0, "takeout": 1.0, "delivery": 0.0},
        0.55,
        mobile=True,
    ),
]
LOC_IDS = [loc.id for loc in LOCATIONS]
LOC = {loc.id: loc for loc in LOCATIONS}


@dataclass(frozen=True)
class Venue:
    id: str
    name: str
    x: float
    y: float
    kind: str  # baseball, football, arena, amphitheater, festival


VENUES: list[Venue] = [
    Venue("ballpark", "Blake Street Ballpark", 0.6, 0.5, "baseball"),
    Venue("stadium", "Platte Field", -1.8, -0.9, "football"),
    Venue("arena", "Larimer Arena", 0.9, -0.5, "arena"),
    Venue("amphitheater", "Red Mesa Amphitheater", -14.0, -16.0, "amphitheater"),
    Venue("civic", "Civic Green", 1.5, -1.0, "festival"),
]
VENUE = {v.id: v for v in VENUES}
# Where the truck parks on days without a big event. Small, steady crowds.
TRUCK_SPOTS = {"office": "Tech Center lot", "brewery": "Brewery row"}


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    unit: str  # purchase unit
    supplier: str
    case_pack: float  # units per case
    cost: float  # dollars per purchase unit
    shelf_life_days: int
    yield_: float  # usable prepared quantity per purchase unit (cooked chicken per raw kg, cooked rice per dry kg)


INGREDIENTS: list[Ingredient] = [
    Ingredient("chicken", "Chicken thigh", "kg", "meats", 10, 7.40, 5, 0.72),
    Ingredient("steak", "Beef flank", "kg", "meats", 8, 15.80, 5, 0.68),
    Ingredient("brisket", "Brisket", "kg", "meats", 12, 10.90, 6, 0.55),
    Ingredient("rice", "Rice, dry", "kg", "dry", 20, 2.20, 180, 2.6),
    Ingredient("beans", "Black beans, canned", "kg", "dry", 12, 3.10, 365, 1.0),
    Ingredient("chickpeas", "Chickpeas, dry", "kg", "dry", 10, 3.40, 365, 2.2),
    Ingredient("chips", "Tortilla chips", "kg", "dry", 5, 6.20, 30, 1.0),
    Ingredient("tea", "Black tea", "kg", "dry", 2, 21.00, 365, 1.0),
    Ingredient("sugar", "Cane sugar", "kg", "dry", 25, 1.20, 365, 1.0),
    Ingredient("tortillas", "Flour tortillas", "each", "bakery", 72, 0.34, 14, 1.0),
    Ingredient("tomatoes", "Tomatoes", "kg", "produce", 11, 3.20, 6, 0.85),
    Ingredient("onions", "Onions", "kg", "produce", 22, 1.40, 21, 0.90),
    Ingredient("green_chile", "Roasted green chiles", "kg", "produce", 10, 6.10, 7, 0.80),
    Ingredient("lettuce", "Romaine", "kg", "produce", 9, 4.00, 5, 0.75),
    Ingredient("corn", "Sweet corn", "kg", "produce", 12, 2.80, 5, 0.60),
    Ingredient("lemons", "Lemons", "kg", "produce", 18, 3.20, 14, 0.40),
    Ingredient("cotija", "Cotija cheese", "kg", "dairy", 5, 12.40, 21, 1.0),
    Ingredient("crema", "Crema", "kg", "dairy", 4.5, 5.20, 14, 1.0),
    Ingredient("milk", "Whole milk", "L", "dairy", 15, 1.10, 10, 1.0),
]
ING = {i.id: i for i in INGREDIENTS}


@dataclass(frozen=True)
class Supplier:
    id: str
    name: str
    lead_days: int
    order_weekdays: tuple[int, ...]  # Monday = 0


SUPPLIERS: list[Supplier] = [
    Supplier("meats", "Front Range Meats", 1, (0, 2, 4)),
    Supplier("produce", "Valley Produce Co.", 1, (0, 1, 2, 3, 4, 5)),
    Supplier("dry", "Plains Dry Goods", 2, (1,)),
    Supplier("bakery", "Larkspur Bakery", 1, (0, 1, 2, 3, 4, 5, 6)),
    Supplier("dairy", "Foothills Dairy", 1, (1, 3, 5)),
]
SUPPLIER = {s.id: s for s in SUPPLIERS}


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    cls: str  # bowl, wrap, side, drink
    price: float
    share: float  # share of a location's units before modifiers
    recipe: dict = field(default_factory=dict)  # ingredient id -> prepared quantity per portion, in purchase units
    launch_days_before_end: int | None = None  # new item: on the menu from this many days before the end
    analog_share: float = 0.0  # cold start: expected share of its class


ITEMS: list[Item] = [
    Item(
        "chicken_bowl",
        "Chicken bowl",
        "bowl",
        12.50,
        0.17,
        {"chicken": 0.15, "rice": 0.20, "beans": 0.08, "tomatoes": 0.04, "lettuce": 0.03, "crema": 0.02},
    ),
    Item(
        "steak_bowl",
        "Steak bowl",
        "bowl",
        14.50,
        0.10,
        {"steak": 0.14, "rice": 0.20, "beans": 0.08, "onions": 0.03, "tomatoes": 0.04, "cotija": 0.015},
    ),
    Item(
        "veggie_bowl",
        "Veggie bowl",
        "bowl",
        11.50,
        0.07,
        {"rice": 0.22, "beans": 0.12, "corn": 0.06, "tomatoes": 0.05, "lettuce": 0.04, "cotija": 0.015},
    ),
    Item(
        "green_chile_bowl",
        "Green chile pork bowl",
        "bowl",
        13.50,
        0.0,
        {"brisket": 0.10, "green_chile": 0.06, "rice": 0.20, "onions": 0.03, "crema": 0.02},
        launch_days_before_end=53,
        analog_share=0.22,
    ),
    Item(
        "chicken_wrap",
        "Chicken wrap",
        "wrap",
        11.00,
        0.11,
        {"chicken": 0.12, "tortillas": 1, "lettuce": 0.04, "tomatoes": 0.03, "crema": 0.02},
    ),
    Item(
        "brisket_wrap",
        "Brisket wrap",
        "wrap",
        13.00,
        0.07,
        {"brisket": 0.12, "tortillas": 1, "onions": 0.03, "green_chile": 0.02},
    ),
    Item(
        "falafel_wrap",
        "Falafel wrap",
        "wrap",
        10.50,
        0.05,
        {"chickpeas": 0.12, "tortillas": 1, "lettuce": 0.04, "tomatoes": 0.04, "crema": 0.02},
    ),
    Item("chips_salsa", "Chips and salsa", "side", 4.00, 0.09, {"chips": 0.09, "tomatoes": 0.06, "onions": 0.02}),
    Item("street_corn", "Street corn", "side", 5.00, 0.05, {"corn": 0.15, "cotija": 0.02, "crema": 0.02}),
    Item("soup", "Black bean soup", "side", 6.00, 0.04, {"beans": 0.18, "onions": 0.04, "tomatoes": 0.04}),
    Item("lemonade", "Lemonade", "drink", 3.75, 0.10, {"lemons": 0.06, "sugar": 0.04}),
    Item("iced_tea", "Iced tea", "drink", 3.25, 0.08, {"tea": 0.008, "sugar": 0.02}),
    Item("horchata", "Horchata", "drink", 4.00, 0.07, {"rice": 0.03, "milk": 0.20, "sugar": 0.03}),
]
ITEM_IDS = [i.id for i in ITEMS]
ITEM = {i.id: i for i in ITEMS}
CLASSES = ["bowl", "wrap", "side", "drink"]
# Average item units per ticket, used to turn units into tickets for the register station.
UNITS_PER_TICKET = 1.7


def raw_per_portion(item: Item) -> dict[str, float]:
    """Purchase units of each ingredient one portion consumes: prepared quantity divided by yield."""
    return {ing: qty / ING[ing].yield_ for ing, qty in item.recipe.items()}


def food_cost(item: Item) -> float:
    return sum(q * ING[ing].cost for ing, q in raw_per_portion(item).items())


def margin(item: Item) -> float:
    return item.price - food_cost(item)


def distance_km(loc: Location, venue: Venue) -> float:
    return ((loc.x - venue.x) ** 2 + (loc.y - venue.y) ** 2) ** 0.5
