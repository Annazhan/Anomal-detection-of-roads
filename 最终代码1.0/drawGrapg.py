import folium
m = folium.Map(location=[39.9042, 116.4074], tiles="OpenStreetMap", zoom_start=10)

for i in range(accident.shape[0]):
    node = [accident["lat"][i],accident["lang"][i]]
    tooltip = "click me! 0.0"
    folium.Marker(location = node, popup='<b>accident!!!</b>', tooltip=tooltip,icon=folium.Icon(color='red')).add_to(m)
filepath = 'data/map.html'
m.save(filepath)