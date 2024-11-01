-- Update the quantity of the item When a new order is placed

CREATE TRIGGER ItemUpdate
AFTER INSERT ON orders
FOR EACH ROW
	UPDATE items
	SET items.quantity = items.quantity - NEW.number
	WHERE items.name = NEW.item_name;