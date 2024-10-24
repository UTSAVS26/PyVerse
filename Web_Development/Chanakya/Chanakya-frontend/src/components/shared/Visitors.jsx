import { useEffect, useState } from "react";

const Visitors = () => {
  const [visits, setVisits] = useState(0);

  useEffect(() => {
    const storedVisits = Number(localStorage.getItem("visitCounter")) || 0;
    setVisits(storedVisits + 1);
  }, []);

  useEffect(() => {
    localStorage.setItem("visitCounter", visits);
  }, [visits]);

  return <p className="mt-3"> Visitors count : {visits}</p>;
};

export default Visitors;
