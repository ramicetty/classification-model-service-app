import React from 'react';
import Tabs from '@material-ui/core/Tabs';
import Paper from '@material-ui/core/Paper';
import Tab from '@material-ui/core/Tab';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import UploadImage from './UploadImage.js'




export default ({option, onSelect, handleChange, onSubmit, imageUrl}) => {

const index = option
const onIndexSelect = (e, index) => onSelect(index)


interface TabPanelProps {
  children?: React.ReactNode;
  index: any;
  value: any;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <Typography
      component="div"
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </Typography>
  );
}

function a11yProps(index: any) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

return (

    <Paper square>
    <Tabs
        value={index}
         indicatorColor="primary"
         textColor="primary"
         onChange={onIndexSelect}>
     <Tab label="Upload Image" {...a11yProps(0)} />
     <Tab label="Image URL"  {...a11yProps(1)} disabled/>
    </Tabs>  
     <TabPanel value={index}  index={0}>
        <UploadImage
        handleChange = {handleChange}
        onSubmit = {onSubmit}
        imageUrl = {imageUrl}

        ></UploadImage>
        </TabPanel>
        <TabPanel value={index} index={1}>
          Item Two
        </TabPanel>   
    </Paper>

    )

}
