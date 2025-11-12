let selected = "";

function placeOrder(foodName) {
  selected = foodName;
  document.getElementById("selectedFood").textContent = foodName;
  document.getElementById("orderForm").style.display = "block";
}

function closeForm() {
  document.getElementById("orderForm").style.display = "none";
}

function submitOrder(event) {
  event.preventDefault();
  document.getElementById("orderForm").style.display = "none";
  document.getElementById("thankYou").classList.remove("hidden");
  setTimeout(() => {
    document.getElementById("thankYou").classList.add("hidden");
  }, 3000);
}
