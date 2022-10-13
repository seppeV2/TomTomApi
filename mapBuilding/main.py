from MapBuilding import initialise_map, add_marker

#our keys
api_key1 = 'm48ctXO8JwRAMDkRxbq2yqtngG3dsO7F'
api_key2 = 'Gcw5ye6iGdXCPqtOllT2idD7OUmOUIG7'
api_key3 = 'ealGbxH7zbd5RKJg94Yz4yvnrB2d0cu1'

#locations =  [longitude, latitude] 
location_amsterdam = [52.377956, 4.897070]
location_brussel = [50.8503396, 4.3517103]


# Save map as TomTom_map
TomTom_map = initialise_map(api_key=api_key1, location=location_brussel, zoom=14, style = "main")
add_marker(TomTom_map, location_brussel, "<b>Brussel Centrum</b>", 'Click here')

    #go to the map.html file and start the live server to view the map
TomTom_map.save('mapBuilding/map.html')
