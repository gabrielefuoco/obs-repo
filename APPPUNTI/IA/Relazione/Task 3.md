magazzino ->centralWarehouse
### Tipi:
- robot
- box
- location
- workstation
- content
- vehicle ->carrier
- place->carrierSlot

### Predicati:
- (filled ?c - content ?b - box)
- (empty ?b - box)
- (has-content ?w - workstation ?c - content)
- (place-vehicle ?p - place ?v - vehicle) ->carrierHasSlot
- (place-available ?p - place) -> carrierSlot-available
- (place-occupied ?p - place) -> carrierSlot-occupied
- (box-loaded ?b - box ?v - vehicle)
- (at ?x - (either robot workstation box content vehicle) ?l - location)
- (need-content ?c - content ?w - workstation)
- (robot-free ?r - robot)
- (connected ?from ?to - location)

### Funzioni:
- (weight ?c - content)
- (vehicle-weight ?v - vehicle) carrier-wheight
- (box-weight ?b - box)
- (path-cost ?r - robot)

### Azioni:
- fill
- charge
- discharge
- move_vehicle
- move_robot
- give_content