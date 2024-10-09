import './App.css';
import Filter from './components/Filter'
import Cards from './components/Cards'
import Spinner from './components/Spinner'
import { apiUrl, filterData } from './data'
import { useEffect, useState } from 'react';
import { toast } from 'react-toastify';

function App() {
  const [courses,setCourses]=useState({});
  const [category,setCategory]=useState(filterData[0].title);
  const [loading,setLoading]=useState(false);

  async function fetchCourses(){
    setLoading(true);
    try {
      let res= await fetch(apiUrl);
      let resData=await res.json();
      setCourses(resData.data);
    } catch (error) {
      toast.error("Error in fetching Courses", { theme: 'dark' });
    }
    setLoading(false);
  }
  useEffect(() => {
    fetchCourses();
  }, [])
  
  return (
    <>
        <div className='heading-container'>
          <h1 className='heading'>Top Courses</h1>
        </div>
        {/* <div className='filter-container'> */}
          <Filter filterData={filterData} category={category} setCategory={setCategory}/>
        {/* </div> */}
        <div className='cards-container'>
          {
            loading ? (<Spinner/>): (<Cards courses={courses} category={category}/>)
          }
        </div>
    </>
  );
}

export default App;
