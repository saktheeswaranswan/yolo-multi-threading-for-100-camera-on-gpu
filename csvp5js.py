import csv

# Open the CSV file
with open('/home/josva/Music/coordinates.csv', newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    # Initialize an empty list to store the points
    points = []
    # Loop through each row in the CSV file
    for row in reader:
        # Create a dictionary with the x and y values
        point = { 'x': int(row[0]), 'y': int(row[1]) }
        # Add the point to the list
        points.append(point)

# Print the JavaScript syntax for the points
print("let points = [")
for point in points:
    print(f"  {{ x: {point['x']}, y: {point['y']} }},")
print("];")

