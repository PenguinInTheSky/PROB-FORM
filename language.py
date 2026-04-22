class Language():
  def __init__(self, constants=[], predicates=[]):
    self.constants = constants
    self.const_to_obj = {}
    self.obj_to_const = {}
        
    self.predicates = predicates
    self.rules = {p: [] for p in predicates}
  
  # get object mapping
  def get_object(self, constant):
    if constant not in self.const_to_obj:
      return None
    return self.const_to_obj[constant]
  
  # get constant mapping
  def get_constant(self, obj) -> str:
    if obj not in self.obj_to_const:
      return None
    return self.obj_to_const[obj]
  
  # get rules for a predicate
  def get_rules(self, predicate):
    if predicate not in self.predicates:
      return None
    return self.rules[predicate]
  
  # add mapping from constant to object, and add the constant to the list of constants if not already present
  def add_constant_mapping(self, constant, obj):
    self.add_constant(constant)
    self.const_to_obj[constant] = obj
    self.obj_to_const[obj] = constant
    
  # add a constant to the list of constants if not already present
  def add_constant(self, constant):
    # make sure constant is unique
    if constant not in self.constants:
      self.constants.append(constant)
    
  # add a rule to the language, and add the predicate to the list of predicates if not already present
  def add_rule(self, predicate, constants):
    self.add_predicate(predicate)
    self.rules[predicate].append(constants)
    
  # add a predicate to the list of predicates if not already present
  def add_predicate(self, predicate):
    if predicate not in self.predicates:
      self.predicates.append(predicate)
      self.rules[predicate] = []
    
  # check if a rule is satisfied given the predicate and the constants
  def check_rule(self, predicate, constants):
    if predicate not in self.predicates:
      return False
    return constants in self.rules[predicate]
  
# test with 2 constants and 1 predicate
if __name__ == "__main__":
  language = Language()
  language.add_constant("o1", "object1")
  language.add_constant("o2", "object2")
  language.add_rule("is_older", ["o1", "o2"])
  
  assert language.check_rule("is_older", ["o1", "o2"]) == True
  assert language.check_rule("is_older", ["o2", "o1"]) == False