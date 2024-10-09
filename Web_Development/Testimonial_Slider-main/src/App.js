import './App.css';
import reviews from './data';
import Testimonials from './components/Testimonials';
function App() {
  return (
    <>
      <div className="container">
      <div className='heading-container'>
        <h1 >Our Testimonials</h1>
        <div className='underline'></div>
      </div>
        <div>
          <Testimonials reviews={reviews}/>
        </div>
      </div>
    </>
  );
}

export default App;
