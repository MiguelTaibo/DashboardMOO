



def disableWidget(label, value):
    box_show_style = "padding: 10px 14px; background: rgb(240, 242, 246); border-radius: 0.25rem;"
    label_div = "<label style='margin-bottom: 7px; font-size: 14px'>"+str(label)+ "</label>"
    value_div = "<div style='"+box_show_style+"'>" + str(value) + "</div>"
    
    return "<div style='margin-bottom: 20px;'>" + label_div + value_div